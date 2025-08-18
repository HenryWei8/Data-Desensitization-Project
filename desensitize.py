import argparse
from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm
import re

PLATE_TRACKS = {}   # track_id -> {"veh":(x,y,w,h), "plate":(x,y,w,h), "age":0}
_NEXT_TRACK_ID = 1

#geometry
def _box_iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1 = max(ax, bx); y1 = max(ay, by)
    x2 = min(ax+aw, bx+bw); y2 = min(ay+ah, by+bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = aw*ah + bw*bh - inter
    return inter/union if union>0 else 0.0

def _pad_frac(box, frac, W, H):
    if frac <= 0: return box
    x,y,w,h = box
    px = int(round(frac*w)); py = int(round(frac*h))
    x -= px; y -= py; w += 2*px; h += 2*py
    x = max(0, min(W-1, x)); y = max(0, min(H-1, y))
    w = max(1, min(W-x, w)); h = max(1, min(H-y, h))
    return (x,y,w,h)

def _temporal_one_plate_per_vehicle(vehicles, plates, img_wh, persist_frames=3, pad_frac=0.10, veh_iou=0.5):
    global PLATE_TRACKS, _NEXT_TRACK_ID
    W, H = img_wh
    perv = {i: [] for i in range(len(vehicles))}
    for (px,py,pw,ph,ps,src) in plates:
        best_i, best = -1, 0.0
        for i,(vx,vy,vw,vh,_,_) in enumerate(vehicles):
            vbox = (vx,vy,vw,vh)
            if center_in(vbox,(px,py,pw,ph)) or _box_iou_xywh(vbox,(px,py,pw,ph))>=0.02:
                iou = _box_iou_xywh(vbox,(px,py,pw,ph))
                if iou>best: best, best_i = iou, i
        if best_i>=0: perv[best_i].append((px,py,pw,ph,ps,src))

    used_tracks = set()
    out = []
    for i,(vx,vy,vw,vh,_,_) in enumerate(vehicles):
        vbox = (vx,vy,vw,vh)
        tid, best = None, 0.0
        for k,t in PLATE_TRACKS.items():
            iou = _box_iou_xywh(t["veh"], vbox)
            if iou >= veh_iou and iou > best:
                best, tid = iou, k
        if tid is None:
            tid = _NEXT_TRACK_ID; _NEXT_TRACK_ID += 1
            PLATE_TRACKS[tid] = {"veh": vbox, "plate": None, "age": persist_frames+1} 
        used_tracks.add(tid)

        if perv[i]:
            cur = max(perv[i], key=lambda t: t[4])
            PLATE_TRACKS[tid]["veh"] = vbox
            PLATE_TRACKS[tid]["plate"] = (cur[0],cur[1],cur[2],cur[3])
            PLATE_TRACKS[tid]["age"] = 0
            out.append(cur)
        else:
            last = PLATE_TRACKS[tid].get("plate")
            age  = PLATE_TRACKS[tid].get("age", 0)
            if last is not None and age < persist_frames:
                px,py,pw,ph = _pad_frac(last, pad_frac, W, H)
                out.append((px,py,pw,ph, 0.99, "persist"))
                PLATE_TRACKS[tid]["veh"] = vbox
                PLATE_TRACKS[tid]["age"] = age + 1
            else:
                PLATE_TRACKS[tid]["veh"] = vbox
                PLATE_TRACKS[tid]["age"] = age + 1
    for k in list(PLATE_TRACKS.keys()):
        if k not in used_tracks:
            PLATE_TRACKS[k]["age"] = PLATE_TRACKS[k].get("age",0)+1
            if PLATE_TRACKS[k]["age"] > persist_frames+2:
                del PLATE_TRACKS[k]
    return out
def iou_xywh(a, b): return _box_iou_xywh(a, b)


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

    if getattr(args, "plate_accept_global", True):
        cands.extend(detect_plates_global(img_bgr, plate_model, args.plate_global_conf, args.plate_iou, args.plate_imgsz))

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

    # geometry + min size
    for (x, y, w, h, s, src) in cands:
        if s < args.plate_min_conf:  # final acceptance threshold 
            continue
        mw = max(args.plate_min_w_px, int(args.plate_min_w_frac * W))
        mh = max(args.plate_min_h_px, int(args.plate_min_h_frac * H))
        if w < mw or h < mh:
            continue
        if w > int(args.plate_max_w_frac * W) or h > int(args.plate_max_h_frac * H):
            continue
        ar = w / max(1.0, h)
        if not (1.1 <= ar <= 10):
            continue
        if args.require_vehicle_overlap and vehicles:
            ok = False
            chosen_v = None
            for (vx, vy, vw, vh, _, _) in vehicles:
                if args.vehicle_pad_frac > 0:
                    px = max(0, vx - int(args.vehicle_pad_frac * vw))
                    py = max(0, vy - int(args.vehicle_pad_frac * vh))
                    pw = min(W - px, vw + int(2 * args.vehicle_pad_frac * vw))
                    ph = min(H - py, vh + int(2 * args.vehicle_pad_frac * vh))
                    vbox = (px, py, pw, ph)
                else:
                    vbox = (vx, vy, vw, vh)

                if center_in(vbox, (x, y, w, h)) or iou_xywh(vbox, (x, y, w, h)) >= args.plate_vehicle_iou:
                    ok = True
                    chosen_v = vbox
                    break
            if not ok:
                continue
            if chosen_v is not None:
                vx, vy, vw, vh = chosen_v
                if (w * h) > args.plate_max_area_wrt_veh * (vw * vh):
                    continue

        kept.append((x, y, w, h, s, src))

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
    if vehicles:
        plates = _temporal_one_plate_per_vehicle(
        vehicles, plates, (img.shape[1], img.shape[0]),
        persist_frames=args.persist_frames,
        pad_frac=args.persist_pad_frac,
        veh_iou=args.track_veh_iou)
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
    gts_by_img_cls = {}
    preds_by_img_cls = {}

    for img_id, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None: continue
        H, W = img.shape[:2]
        rel = f.relative_to(in_root)
        gt_path = gt_root / rel.with_suffix(".txt")
        gts = _yolo_txt_to_abs(gt_path, W, H)
        for (x, y, w, h, cls) in gts:
            gts_by_img_cls.setdefault((img_id, cls), []).append((x, y, w, h))

        preds = _predict_for_eval(img, args, yunet, veh_model, plate_model)
        for (cls, sc, box) in preds:
            preds_by_img_cls.setdefault((img_id, cls), []).append((sc, box))

    classes = sorted(set(c for (_, c) in gts_by_img_cls) | set(c for (_, c) in preds_by_img_cls))
    if args.eval_keep_classes:
        classes = [c for c in classes if c in set(args.eval_keep_classes)]

    print("== Evaluation ==")
    overall_tp = overall_fn = overall_tn = 0
    for cls in classes:
        tp = fn = tn = 0
        for img_id, f in enumerate(files):
            gt_list = gts_by_img_cls.get((img_id, cls), [])
            pr_list = preds_by_img_cls.get((img_id, cls), [])

            gt_pos = len(gt_list) > 0
            if gt_pos:
                matched_gt = 0
                if pr_list:
                    for gt in gt_list:
                        if any(
                            (_coverage(gt, pbox) >= args.cov_thresh) or
                            (iou_xywh(gt, pbox) >= args.iou_thresh) or
                            _center_in(gt, pbox)
                            for (_, pbox) in pr_list
                        ):
                            matched_gt += 1
                frac = matched_gt / float(len(gt_list)) if gt_list else 0.0
                if frac >= args.frame_ok_thresh:
                    tp += 1
                else:
                    fn += 1
            else:
                tn += 1


        overall_tp += tp; overall_fn += fn; overall_tn += tn
        prec = tp / max(1, tp)  
        rec  = tp / max(1, tp + fn)
        acc  = (tp + tn) / max(1, tp + tn + fn)
        f1   = (2 * tp) / max(1, 2 * tp + fn)
        print(f"class {cls}: acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")

    overall_prec = overall_tp / max(1, overall_tp)  
    overall_rec  = overall_tp / max(1, overall_tp + overall_fn)
    overall_acc  = (overall_tp + overall_tn) / max(1, overall_tp + overall_tn + overall_fn)
    overall_f1   = (2 * overall_tp) / max(1, 2 * overall_tp + overall_fn)
    print(f"overall: acc={overall_acc:.4f}  prec={overall_prec:.4f}  rec={overall_rec:.4f}  f1={overall_f1:.4f}")
    return 0.0, {}

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
    ap.add_argument("--vehicle-imgsz", type=int,   default=1536)
    ap.add_argument("--vehicle-pad-frac", type=float, default=0.12)


    ap.add_argument("--plate-model",        type=str,   default="weights/plate-v1n.pt")
    ap.add_argument("--plate-probe-conf",   type=float, default=0.001)
    ap.add_argument("--plate-global-conf",  type=float, default=0.35)
    ap.add_argument("--plate-iou",          type=float, default=0.60)
    ap.add_argument("--plate-imgsz",        type=int,   default=2304)

    ap.add_argument("--plate-max-w-frac",   type=float, default=0.30)  # max width vs image
    ap.add_argument("--plate-max-h-frac",   type=float, default=0.20)  # max height vs image
    ap.add_argument("--plate-max-area-wrt-veh", type=float, default=0.25)  # max area vs vehicle box

    ap.add_argument("--plate-min-w-px",   type=int,   default=10)
    ap.add_argument("--plate-min-h-px",   type=int,   default=6)
    ap.add_argument("--plate-min-w-frac", type=float, default=0.006)
    ap.add_argument("--plate-min-h-frac", type=float, default=0.004)
    ap.add_argument("--plate-min-conf",   type=float, default=0.04)   # final acceptance threshold

    ap.add_argument("--require-vehicle-overlap", dest="require_vehicle_overlap", action="store_true", default=True)
    ap.add_argument("--plate-vehicle-iou", type=float, default=0.02)
    ap.add_argument("--plate-merge-iou",   type=float, default=0.5)
    ap.add_argument("--one-per-vehicle",   action="store_true", default=False)

    ap.add_argument("--viz-all-plates", action="store_true")
    ap.add_argument("--persist-frames", type=int, default=3)
    ap.add_argument("--persist-pad-frac", type=float, default=0.10)
    ap.add_argument("--track-veh-iou", type=float, default=0.5)

    ap.add_argument("--blur", choices=["gaussian", "pixelate"], default="pixelate")
    ap.add_argument("--kernel",     type=int, default=None)
    ap.add_argument("--pixel-size", type=int, default=8)

    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--frame-ok-thresh", type=float, default=0.5)
    ap.add_argument("--eval-keep-classes", type=str, default=None)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--gt-root", type=str, default=None)
    ap.add_argument("--cov-thresh", type=float, default=0.30)
    ap.add_argument("--iou-thresh", type=float, default=0.10)

    args = ap.parse_args()
    global PLATE_TRACKS, _NEXT_TRACK_ID
    PLATE_TRACKS.clear(); _NEXT_TRACK_ID = 1
    if args.eval_keep_classes:
        args.eval_keep_classes = [int(x) for x in args.eval_keep_classes.split(",")]
    else:
        args.eval_keep_classes = None

    yunet       = None if args.no_faces    else load_yunet(args.face_score_thres, args.face_nms_thres, args.face_imgsz)
    veh_model   = None if args.no_vehicles else load_yolo(args.vehicle_model)
    plate_model = None if args.no_plates   else load_yolo(args.plate_model)

    in_path, out_root = Path(args.input), Path(args.output)
    if in_path.is_file():
        files = [in_path]; outs = [out_root if out_root.suffix else out_root / in_path.name]
    else:
        def _nat_key(p):  # natural sort: frame_2.png before frame_10.png
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(p))]

        files = sorted(in_path.rglob(args.glob), key=_nat_key)
        outs  = [out_root / f.relative_to(in_path) for f in files]
    if not files:
        print("No input images found."); return

    for f, outf in tqdm(list(zip(files, outs)), total=len(files), desc="Processing"):
        ok, err = process_image(f, outf, args, yunet, veh_model, plate_model)
        if not ok: tqdm.write(f"[WARN] {f}: {err}")

    if args.eval and in_path.is_dir():
        gt_root = Path(args.gt_root) if args.gt_root else in_path.parents[1] / "annotated"
        _eval_dataset(args, files, in_path, gt_root, yunet, veh_model, plate_model)


if __name__ == "__main__":
    main()
