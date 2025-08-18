import argparse
from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm

#geometry
def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx); y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw); y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / float(union) if union > 0 else 0.0


def ensure_parent_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def clamp_box(x, y, w, h, W, H):
    x = max(0, min(W - 1, x)); y = max(0, min(H - 1, y))
    w = max(1, min(W - x, w)); h = max(1, min(H - y, h))
    return x, y, w, h


def center_in(big, small):
    cx = small[0] + small[2] / 2.0
    cy = small[1] + small[3] / 2.0
    return (big[0] <= cx <= big[0] + big[2]) and (big[1] <= cy <= big[1] + big[3])


# blur

def draw_box(img, box, label, conf=None, color=(0, 255, 0), thick=2):
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thick)
    text = f"{label} {conf:.2f}" if (conf is not None) else label
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y2 = max(th + 4, y)
    cv2.rectangle(img, (x, y2 - th - 4), (x + tw + 6, y2), color, -1)
    cv2.putText(img, text, (x + 3, y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def _natural_composite(roi, blur, feather=10, round_frac=0.18):
    h, w = roi.shape[:2]
    r = max(1, min(int(round(round_frac * min(w, h))), min(w, h) // 2))
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
    cv2.circle(mask, (r, r), r, 255, -1)
    cv2.circle(mask, (w - r, r), r, 255, -1)
    cv2.circle(mask, (r, h - r), r, 255, -1)
    cv2.circle(mask, (w - r, h - r), r, 255, -1)
    if feather > 0:
        f = int(feather * 2 + 1)
        mask = cv2.GaussianBlur(mask, (f, f), 0)
    a = (mask.astype(np.float32) / 255.0)[..., None]
    return (blur.astype(np.float32) * a + roi.astype(np.float32) * (1 - a)).astype(np.uint8)


def gaussian_blur_roi(img, x, y, w, h, k=None):
    H, W = img.shape[:2]
    x, y, w, h = clamp_box(x, y, w, h, W, H)
    roi = img[y:y + h, x:x + w]
    if roi.size == 0: return
    if k is None: k = max(5, (min(w, h) // 6) | 1)
    if k % 2 == 0: k += 1
    blur = cv2.GaussianBlur(roi, (k, k), 0)
    img[y:y + h, x:x + w] = _natural_composite(roi, blur, feather=10, round_frac=0.18)


def pixelate_roi(img, x, y, w, h, pix=12):
    H, W = img.shape[:2]
    x, y, w, h = clamp_box(x, y, w, h, W, H)
    roi = img[y:y + h, x:x + w]
    if roi.size == 0: return
    rw, rh = roi.shape[1], roi.shape[0]
    ws = max(1, rw // max(1, pix)); hs = max(1, rh // max(1, pix))
    small = cv2.resize(roi, (ws, hs), interpolation=cv2.INTER_LINEAR)
    blur = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    img[y:y + h, x:x + w] = _natural_composite(roi, blur, feather=10, round_frac=0.18)


# detectors

def _ensure_yunet_path() -> str:
    w = Path("weights/face_detection_yunet_2023mar.onnx")
    if not w.exists():
        import ssl, urllib.request, certifi
        w.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=ctx) as r, open(w, "wb") as f:
            f.write(r.read())
    return str(w)


def load_yunet(score_thres=0.6, nms_thres=0.3, size=320):
    if not hasattr(cv2, "FaceDetectorYN_create"):
        return None
    return cv2.FaceDetectorYN_create(
        _ensure_yunet_path(), "", (size, size),
        score_threshold=score_thres, nms_threshold=nms_thres, top_k=5000
    )


def detect_faces(img_bgr, yunet):
    if yunet is None: return []
    H, W = img_bgr.shape[:2]
    yunet.setInputSize((W, H))
    _, dets = yunet.detect(img_bgr)
    out = []
    if dets is not None:
        for d in dets:
            x, y, w, h = map(int, d[:4])
            s = max(0.0, min(1.0, float(d[4])))
            out.append((x, y, w, h, s))
    return out


def load_yolo(model_path: str):
    from ultralytics import YOLO
    m = YOLO(model_path)
    _ = m.predict(source=np.zeros((10, 10, 3), dtype=np.uint8), conf=0.01, verbose=False)
    return m


def yolo_detect(img_bgr, model, conf=0.25, iou=0.5, imgsz=640, name_allow=None):
    H, W = img_bgr.shape[:2]
    out = []
    for r in model.predict(source=img_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False, agnostic_nms=True):
        names = r.names
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            c = float(b.conf[0]); cls = int(b.cls[0])
            name = names.get(cls, str(cls)).lower()
            if (name_allow is None) or (name in name_allow):
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(W - 1, x2); y2 = min(H - 1, y2)
                if x2 > x1 and y2 > y1:
                    out.append((x1, y1, x2 - x1, y2 - y1, c, name))
    return out


def detect_plates_global(img_bgr, plate_model, conf, iou, imgsz):
    H, W = img_bgr.shape[:2]
    out = []
    res = plate_model.predict(source=img_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False, agnostic_nms=True)
    for r in res:
        names = r.names
        for b in r.boxes:
            name = names.get(int(b.cls[0]), "?").lower()
            if ("plate" not in name) and ("license" not in name): continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            s = float(b.conf[0])
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(W - 1, x2); y2 = min(H - 1, y2)
            if x2 > x1 and y2 > y1:
                out.append((x1, y1, x2 - x1, y2 - y1, s, "global"))
    return out


def collect_plate_candidates(img_bgr, plate_model, vehicles, args):
    H, W = img_bgr.shape[:2]
    cands = []

    # global
    if getattr(args, "plate_accept_global", True):
        cands.extend(detect_plates_global(img_bgr, plate_model, args.plate_global_conf, args.plate_iou, args.plate_imgsz))

    # per-vehicle crops (probe at very low conf to maximize recall)
    for vi, (vx, vy, vw, vh, _, vname) in enumerate(vehicles):
        vx, vy, vw, vh = clamp_box(vx, vy, vw, vh, W, H)
        crop = img_bgr[vy:vy + vh, vx:vx + vw]
        res = plate_model.predict(source=crop, conf=args.plate_probe_conf, iou=args.plate_iou, imgsz=args.plate_imgsz, verbose=False, agnostic_nms=True)
        for r in res:
            names = r.names
            for b in r.boxes:
                name = names.get(int(b.cls[0]), "?").lower()
                if ("plate" not in name) and ("license" not in name): continue
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                s = float(b.conf[0])
                x1 += vx; y1 += vy; x2 += vx; y2 += vy
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(W - 1, x2); y2 = min(H - 1, y2)
                if x2 <= x1 or y2 <= y1: continue
                cands.append((x1, y1, x2 - x1, y2 - y1, s, f"veh{vi}"))

    return cands


def filter_and_dedup(cands, vehicles, img_wh, args):
    W, H = img_wh
    kept = []

    # light geometry + min size
    for (x, y, w, h, s, src) in cands:
        if s < args.plate_min_conf:  # final acceptance threshold 
            continue
        mw = max(args.plate_min_w_px, int(args.plate_min_w_frac * W))
        mh = max(args.plate_min_h_px, int(args.plate_min_h_frac * H))
        if w < mw or h < mh:
            continue
        ar = w / max(1.0, h)
        if not (1.1 <= ar <= 9.0):   # very forgiving
            continue

        if args.require_vehicle_overlap and vehicles:
            ok = False
            for (vx, vy, vw, vh, _, _) in vehicles:
                vbox = (vx, vy, vw, vh)
                if center_in(vbox, (x, y, w, h)) or iou_xywh(vbox, (x, y, w, h)) >= args.plate_vehicle_iou:
                    ok = True; break
            if not ok:
                continue

        kept.append((x, y, w, h, s, src))

    # dedup by IoU, keep highest score
    kept.sort(key=lambda t: t[4], reverse=True)
    final = []
    for cand in kept:
        if all(iou_xywh((cand[0], cand[1], cand[2], cand[3]), (b[0], b[1], b[2], b[3])) < args.plate_merge_iou for b in final):
            final.append(cand)

    if args.one_per_vehicle and vehicles:
        perv = []
        for i, v in enumerate(vehicles):
            vx, vy, vw, vh, _, _ = v
            best, best_s = None, -1.0
            for (x, y, w, h, s, src) in final:
                if center_in((vx, vy, vw, vh), (x, y, w, h)) or iou_xywh((vx, vy, vw, vh), (x, y, w, h)) >= 0.02:
                    if s > best_s: best_s, best = s, (x, y, w, h, s, src)
            if best: perv.append(best)
        return perv

    return final

def process_image(path_in, path_out, args, yunet, veh_model, plate_model):
    img = cv2.imread(str(path_in))
    if img is None: return False, f"Cannot read {path_in}"
    H, W = img.shape[:2]

    faces = [] if args.no_faces else detect_faces(img, yunet)
    vehicles = [] if args.no_vehicles else yolo_detect(
        img, veh_model, conf=args.vehicle_conf, iou=args.vehicle_iou, imgsz=args.vehicle_imgsz,
        name_allow=["car", "truck", "bus", "motorcycle", "motorbike"]
    )

    all_plate_cands = [] if args.no_plates else collect_plate_candidates(img, plate_model, vehicles, args)
    plates = [] if args.no_plates else filter_and_dedup(all_plate_cands, vehicles, (W, H), args)

    if args.debug:
        print(f"[{path_in.name}] veh={len(vehicles)} cand={len(all_plate_cands)} kept={len(plates)} faces={len(faces)}")

    if args.mode == "boxes":
        for (x, y, w, h, s, _) in vehicles: draw_box(img, (x, y, w, h), "VEHICLE", s, (0, 255, 0), 2)
        for (x, y, w, h, s, src) in plates: draw_box(img, (x, y, w, h), "PLATE", s if s >= 0.05 else None, (0, 180, 255), 2)
        if args.viz_all_plates:
            for (x, y, w, h, s, src) in all_plate_cands:
                draw_box(img, (x, y, w, h), f"ALL {s:.2f}", s, (255, 0, 255), 2)
    else:
        rois = []
        if not args.no_plates: rois += [(x, y, w, h) for (x, y, w, h, _, _) in plates]
        if not args.no_faces:  rois += [(x, y, w, h) for (x, y, w, h, _) in faces]
        k = args.kernel if (args.kernel is None or args.kernel % 2 == 1) else args.kernel + 1
        for (x, y, w, h) in rois:
            if args.blur == "pixelate": pixelate_roi(img, x, y, w, h, args.pixel_size)
            else: gaussian_blur_roi(img, x, y, w, h, k)

    ensure_parent_dir(path_out)
    ok = cv2.imwrite(str(path_out), img)
    return ok, None if ok else f"Failed to write {path_out}"

def _yolo_txt_to_abs(txt_path: Path, W: int, H: int):
    if not txt_path.exists(): return []
    out = []
    for line in txt_path.read_text().strip().splitlines():
        if not line.strip(): continue
        parts = line.split()
        cls = int(parts[0]); cx, cy, w, h = map(float, parts[1:5])
        ww = max(1, int(round(w * W))); hh = max(1, int(round(h * H)))
        x = int(round(cx * W - ww / 2)); y = int(round(cy * H - hh / 2))
        x, y, ww, hh = clamp_box(x, y, ww, hh, W, H)
        out.append((x, y, ww, hh, cls))
    return out


def _predict_for_eval(img, args, yunet, veh_model, plate_model):
    H, W = img.shape[:2]

    faces = [] if args.no_faces else detect_faces(img, yunet)

    vehicles = [] if args.no_vehicles else yolo_detect(
        img, veh_model,
        conf=args.vehicle_conf, iou=args.vehicle_iou, imgsz=args.vehicle_imgsz,
        name_allow=["car", "truck", "bus", "motorcycle", "motorbike"]
    )

    all_cands = [] if args.no_plates else collect_plate_candidates(img, plate_model, vehicles, args)
    plates    = [] if args.no_plates else filter_and_dedup(all_cands, vehicles, (W, H), args)

    preds = []
    for (x, y, w, h, s, _) in plates:
        preds.append((1, float(s), (x, y, w, h)))  # class 1 = plate
    for (x, y, w, h, s) in faces:
        preds.append((0, float(s), (x, y, w, h)))  # class 0 = face
    return preds



def _coverage(gt, pr):
    inter = max(0, min(gt[0] + gt[2], pr[0] + pr[2]) - max(gt[0], pr[0])) * \
            max(0, min(gt[1] + gt[3], pr[1] + pr[3]) - max(gt[1], pr[1]))
    return inter / float(max(1, gt[2] * gt[3]))


def _center_in(gt, pr):
    cx = pr[0] + pr[2] / 2.0; cy = pr[1] + pr[3] / 2.0
    return (gt[0] <= cx <= gt[0] + gt[2]) and (gt[1] <= cy <= gt[1] + gt[3])


def _eval_dataset(args, files, in_root, gt_root, yunet, veh_model, plate_model):
    all_preds = []; gts_by_img_cls = {}
    for img_id, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None: continue
        H, W = img.shape[:2]
        rel = f.relative_to(in_root)
        gt_path = gt_root / rel.with_suffix(".txt")
        gts = _yolo_txt_to_abs(gt_path, W, H)
        for (x, y, w, h, cls) in gts:
            gts_by_img_cls.setdefault((img_id, cls), []).append({"box": (x, y, w, h), "m": False})
        preds = _predict_for_eval(img, args, yunet, veh_model, plate_model)
        for (cls, sc, box) in preds:
            all_preds.append((img_id, cls, sc, box))
    all_preds.sort(key=lambda t: t[2], reverse=True)

    def is_match(gtb, prb):
        if _coverage(gtb, prb) >= args.cov_thresh: return True
        if iou_xywh(gtb, prb) >= args.iou_thresh: return True
        if _center_in(gtb, prb): return True
        return False

    classes = sorted(set(c for _, c, _, _ in all_preds) | set(c for (_, c) in gts_by_img_cls))
    results = {}; map_vals = []
    for cls in classes:
        preds = [(img, sc, box) for (img, c, sc, box) in all_preds if c == cls]
        n_gt = sum(len(v) for (k, v) in gts_by_img_cls.items() if k[1] == cls)
        if n_gt == 0: continue
        matched = {k: [False] * len(v) for (k, v) in gts_by_img_cls.items() if k[1] == cls}
        tp, fp = [], []
        for (img, sc, pbox) in preds:
            key = (img, cls); jbest, best = -1, -1.0
            if key in gts_by_img_cls:
                for j, gt in enumerate(gts_by_img_cls[key]):
                    if matched[key][j]: continue
                    if is_match(gt["box"], pbox):
                        iov = iou_xywh(gt["box"], pbox)
                        if iov > best: best, jbest = iov, j
            if jbest >= 0:
                matched[key][jbest] = True; tp.append(1); fp.append(0)
            else:
                tp.append(0); fp.append(1)
        tp = np.array(tp); fp = np.array(fp)
        if tp.size == 0:
            results[cls] = {"AP": 0.0, "precision": [], "recall": []}; continue
        tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
        rec = tp_c / float(n_gt)
        prec = tp_c / np.maximum(1, tp_c + fp_c)
        mprec = np.maximum.accumulate(prec[::-1])[::-1]
        ap = np.trapz(mprec, rec)
        results[cls] = {"AP": float(ap), "precision": rec.tolist(), "recall": rec.tolist()}
        map_vals.append(ap)
    mAP = float(np.mean(map_vals)) if map_vals else 0.0
    return mAP, results


def main():
    ap = argparse.ArgumentParser("Vehicles → plates: robust keep-all or one-per-vehicle with generous gates")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--glob", default="*.[Jj][Pp]*[Gg]")
    ap.add_argument("--mode", choices=["boxes", "blur"], default="boxes")
    ap.add_argument("--no-faces", action="store_true")
    ap.add_argument("--no-vehicles", action="store_true")
    ap.add_argument("--no-plates", action="store_true")
    ap.add_argument("--plate-accept-global", action="store_true", default=True)
    ap.add_argument("--no-plate-accept-global", dest="plate_accept_global", action="store_false")

    ap.add_argument("--face-score-thres", type=float, default=0.50)
    ap.add_argument("--face-nms-thres",   type=float, default=0.45)
    ap.add_argument("--face-imgsz",       type=int,   default=800)

    ap.add_argument("--vehicle-model", type=str, default="yolov8n.pt")
    ap.add_argument("--vehicle-conf",  type=float, default=0.25)
    ap.add_argument("--vehicle-iou",   type=float, default=0.60)
    ap.add_argument("--vehicle-imgsz", type=int,   default=1024)

    ap.add_argument("--plate-model",        type=str,   default="weights/plate-v1n.pt")
    ap.add_argument("--plate-probe-conf",   type=float, default=0.001)
    ap.add_argument("--plate-global-conf",  type=float, default=0.35)
    ap.add_argument("--plate-iou",          type=float, default=0.60)
    ap.add_argument("--plate-imgsz",        type=int,   default=2048)

    ap.add_argument("--plate-min-w-px",   type=int,   default=10)
    ap.add_argument("--plate-min-h-px",   type=int,   default=6)
    ap.add_argument("--plate-min-w-frac", type=float, default=0.006)
    ap.add_argument("--plate-min-h-frac", type=float, default=0.004)
    ap.add_argument("--plate-min-conf",   type=float, default=0.05)   # final acceptance threshold

    ap.add_argument("--require-vehicle-overlap", dest="require_vehicle_overlap", action="store_true", default=True)
    ap.add_argument("--plate-vehicle-iou", type=float, default=0.02)
    ap.add_argument("--plate-merge-iou",   type=float, default=0.5)
    ap.add_argument("--one-per-vehicle",   action="store_true", default=False)

    ap.add_argument("--viz-all-plates", action="store_true")

    ap.add_argument("--blur", choices=["gaussian", "pixelate"], default="pixelate")
    ap.add_argument("--kernel",     type=int, default=None)
    ap.add_argument("--pixel-size", type=int, default=14)

    ap.add_argument("--debug", action="store_true")

    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--gt-root", type=str, default=None)
    ap.add_argument("--cov-thresh", type=float, default=0.30)
    ap.add_argument("--iou-thresh", type=float, default=0.10)

    args = ap.parse_args()

    yunet       = None if args.no_faces    else load_yunet(args.face_score_thres, args.face_nms_thres, args.face_imgsz)
    veh_model   = None if args.no_vehicles else load_yolo(args.vehicle_model)
    plate_model = None if args.no_plates   else load_yolo(args.plate_model)

    in_path, out_root = Path(args.input), Path(args.output)
    if in_path.is_file():
        files = [in_path]; outs = [out_root if out_root.suffix else out_root / in_path.name]
    else:
        files = list(in_path.rglob(args.glob))
        outs  = [out_root / f.relative_to(in_path) for f in files]
    if not files:
        print("No input images found."); return

    for f, outf in tqdm(list(zip(files, outs)), total=len(files), desc="Processing"):
        ok, err = process_image(f, outf, args, yunet, veh_model, plate_model)
        if not ok: tqdm.write(f"[WARN] {f}: {err}")

    if args.eval and in_path.is_dir():
        gt_root = Path(args.gt_root) if args.gt_root else in_path.parents[1] / "annotated"
        mAP, res = _eval_dataset(args, files, in_path, gt_root, yunet, veh_model, plate_model)
        print("== Evaluation ==")
        for cls, v in sorted(res.items()):
            print(f"class {cls} AP: {v['AP']:.4f}")
        print(f"mAP: {mAP:.4f}")


if __name__ == "__main__":
    main()
