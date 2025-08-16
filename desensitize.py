import argparse
from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm


def in_lower_vehicle(y, h, vy, vh, low_frac=0.35, high_frac=0.95):
    cy = y + 0.5*h
    return (vy + low_frac*vh) <= cy <= (vy + high_frac*vh)

def iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0: return 0.0
    return inter / float(aw * ah + bw * bh - inter)

def detect_plates_global(img_bgr, plate_model, conf, iou, imgsz):
    H, W = img_bgr.shape[:2]
    out = []
    res = plate_model.predict(source=img_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    for r in res:
        names = r.names
        for b in r.boxes:
            name = names.get(int(b.cls[0]), "?").lower()
            if "plate" not in name and "license" not in name:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            s = float(b.conf[0])
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(W - 1, x2); y2 = min(H - 1, y2)
            if x2 > x1 and y2 > y1:
                out.append((x1, y1, x2 - x1, y2 - y1, s))
    return out


def ensure_parent_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def clamp_box(x, y, w, h, W, H):
    x = max(0, min(W - 1, x)); y = max(0, min(H - 1, y))
    w = max(1, min(W - x, w)); h = max(1, min(H - y, h))
    return x, y, w, h


def draw_box(img, box, label, conf=None, color=(0, 255, 0), thick=2):
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thick)
    text = f"{label} {conf:.2f}" if conf is not None else label
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y2 = max(th + 4, y)
    cv2.rectangle(img, (x, y2 - th - 4), (x + tw + 6, y2), color, -1)
    cv2.putText(img, text, (x + 3, y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


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
    return cv2.FaceDetectorYN_create(
        _ensure_yunet_path(), "", (size, size),
        score_threshold=score_thres, nms_threshold=nms_thres, top_k=5000
    )


def detect_faces(img_bgr, yunet):
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
    for r in model.predict(source=img_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False):
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


def gaussian_blur_roi(img, x, y, w, h, k=None):
    roi = img[y:y + h, x:x + w]
    if roi.size == 0: return
    if k is None:
        k = max(5, (min(w, h) // 6) | 1)
    if k % 2 == 0: k += 1
    img[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (k, k), 0)


def pixelate_roi(img, x, y, w, h, pix=12):
    roi = img[y:y + h, x:x + w]
    if roi.size == 0: return
    ws = max(1, w // max(1, pix)); hs = max(1, h // max(1, pix))
    small = cv2.resize(roi, (ws, hs), interpolation=cv2.INTER_LINEAR)
    img[y:y + h, x:x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def assign_one_plate_per_vehicle(
    img_bgr, plate_model, vehicles,
    probe_conf, inveh_conf, iou, imgsz,
    leftmost_bonus, bl_allow, bl_x_frac, bl_y_frac,
    min_w_px, min_h_px, min_w_frac, min_h_frac
):
    H, W = img_bgr.shape[:2]
    chosen = []
    for (vx, vy, vw, vh, _, _) in vehicles:
        vx, vy, vw, vh = clamp_box(vx, vy, vw, vh, W, H)
        crop = img_bgr[vy:vy + vh, vx:vx + vw]
        res = plate_model.predict(source=crop, conf=probe_conf, iou=iou, imgsz=imgsz, verbose=False)
        cands = []
        for r in res:
            names = r.names
            for b in r.boxes:
                name = names.get(int(b.cls[0]), "?").lower()
                if "plate" not in name and "license" not in name: continue
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                s = float(b.conf[0])
                x1 += vx; y1 += vy; x2 += vx; y2 += vy
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(W - 1, x2); y2 = min(H - 1, y2)
                if x2 <= x1 or y2 <= y1: continue
                w = x2 - x1; h = y2 - y1
                mw = max(min_w_px, int(min_w_frac * vw))
                mh = max(min_h_px, int(min_h_frac * vh))
                if w < mw or h < mh: continue
                w = x2 - x1; h = y2 - y1
                if not in_lower_vehicle(y1, h, vy, vh): continue
                cands.append((x1, y1, w, h, s))
        if not cands: continue
        leftmost_x = min(c[0] for c in cands)
        strong = [c for c in cands if c[4] >= inveh_conf]
        pool = strong
        if not pool and bl_allow:
            bx = vx + bl_x_frac * vw
            by = vy + vh - bl_y_frac * vh
            pool = []
            for (x, y, w, h, s) in cands:
                cx, cy = x + w / 2, y + h / 2
                if cx <= bx and cy >= by:
                    pool.append((x, y, w, h, s))
        if not pool: continue
        best, best_sc = None, -1.0
        for (x, y, w, h, s) in pool:
            sc = s + (leftmost_bonus if x == leftmost_x else 0.0)
            if sc > best_sc:
                best_sc, best = sc, (x, y, w, h, s)
        if best: chosen.append(best)
    return chosen


def process_image(path_in, path_out, args, yunet, veh_model, plate_model):
    img = cv2.imread(str(path_in))
    if img is None: return False, f"Cannot read {path_in}"
    faces = [] if args.no_faces else detect_faces(img, yunet)
    vehicles = [] if args.no_vehicles else yolo_detect(
        img, veh_model,
        conf=args.vehicle_conf, iou=args.vehicle_iou, imgsz=args.vehicle_imgsz,
        name_allow=["car", "truck", "bus", "motorcycle", "motorbike"]
    )
    plates = [] if args.no_plates else assign_one_plate_per_vehicle(
        img, plate_model, vehicles,
        args.plate_probe_conf, args.plate_inveh_conf,
        args.plate_iou, args.plate_imgsz, args.plate_leftmost_bonus,
        args.plate_override_bottomleft, args.plate_blx_frac, args.plate_bly_frac,
        args.plate_min_w_px, args.plate_min_h_px, args.plate_min_w_frac, args.plate_min_h_frac
    )
    if (not args.no_plates) and args.plate_accept_global:
        gplates = detect_plates_global(img, plate_model, args.plate_global_conf, args.plate_iou, args.plate_imgsz)
        for gx, gy, gw, gh, gs in gplates:
            if any(iou_xywh((gx, gy, gw, gh), (x, y, w, h)) >= args.plate_dup_iou
                   for (x, y, w, h, _) in plates):
                continue
            plates.append((gx, gy, gw, gh, gs))
    if args.mode == "boxes":
        for (x, y, w, h, s, _) in vehicles: draw_box(img, (x, y, w, h), "VEHICLE", s, (0, 255, 0), 2)
        for (x, y, w, h, s)     in plates:   draw_box(img, (x, y, w, h), "PLATE",   s, (0, 180, 255), 2)
        for (x, y, w, h, s)     in faces:    draw_box(img, (x, y, w, h), "FACE",    s, (0, 255, 255), 2)
    else:
        rois = []
        if not args.no_plates: rois += [(x, y, w, h) for (x, y, w, h, _) in plates]
        if not args.no_faces:  rois += [(x, y, w, h) for (x, y, w, h, _) in faces]
        k = args.kernel if (args.kernel is None or args.kernel % 2 == 1) else args.kernel + 1
        for (x, y, w, h) in rois:
            if args.blur == "pixelate": pixelate_roi(img, x, y, w, h, args.pixel_size)
            else: gaussian_blur_roi(img, x, y, w, h, k)
    ensure_parent_dir(path_out)
    ok = cv2.imwrite(str(path_out), img)
    return ok, None if ok else f"Failed to write {path_out}"


def main():
    ap = argparse.ArgumentParser("Vehicles → pick at most one plate per vehicle (bottom-left fallback)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--glob", default="*.[Jj][Pp]*[Gg]")
    ap.add_argument("--mode", choices=["boxes", "blur"], default="boxes")
    ap.add_argument("--no-faces", action="store_true")
    ap.add_argument("--no-vehicles", action="store_true")
    ap.add_argument("--no-plates", action="store_true")
    ap.add_argument("--face-score-thres", type=float, default=0.50)
    ap.add_argument("--face-nms-thres",   type=float, default=0.45)
    ap.add_argument("--face-imgsz",       type=int,   default=800)
    ap.add_argument("--vehicle-model", type=str, default="yolov8n.pt")
    ap.add_argument("--vehicle-conf",  type=float, default=0.25)
    ap.add_argument("--vehicle-iou",   type=float, default=0.50)
    ap.add_argument("--vehicle-imgsz", type=int,   default=1024)
    ap.add_argument("--plate-min-w-px",   type=int,   default=18)
    ap.add_argument("--plate-min-h-px",   type=int,   default=8)
    ap.add_argument("--plate-min-w-frac", type=float, default=0.03)
    ap.add_argument("--plate-min-h-frac", type=float, default=0.02)
    ap.add_argument("--plate-accept-global", action="store_true", default=True)
    ap.add_argument("--plate-global-conf", type=float, default=0.50)
    ap.add_argument("--plate-dup-iou", type=float, default=0.5)
    ap.add_argument("--plate-model",        type=str,   default="weights/plate-v1n.pt")
    ap.add_argument("--plate-inveh-conf",   type=float, default=0.40)
    ap.add_argument("--plate-probe-conf",   type=float, default=0.001)
    ap.add_argument("--plate-iou",          type=float, default=0.50)
    ap.add_argument("--plate-imgsz",        type=int,   default=1536)
    ap.add_argument("--plate-leftmost-bonus", type=float, default=0.03)
    ap.add_argument("--plate-override-bottomleft", action="store_true", default=True)
    ap.add_argument("--plate-blx-frac", type=float, default=0.15)
    ap.add_argument("--plate-bly-frac", type=float, default=0.20)
    ap.add_argument("--blur", choices=["gaussian", "pixelate"], default="pixelate")
    ap.add_argument("--kernel",     type=int, default=None)
    ap.add_argument("--pixel-size", type=int, default=14)
    args = ap.parse_args()
    yunet       = None if args.no_faces    else load_yunet(args.face_score_thres, args.face_nms_thres, args.face_imgsz)
    veh_model   = None if args.no_vehicles else load_yolo(args.vehicle_model)
    plate_model = None if args.no_plates   else load_yolo(args.plate_model)
    in_path, out_root = Path(args.input), Path(args.output)
    if in_path.is_file():
        files = [in_path]
        outs  = [out_root if out_root.suffix else out_root / in_path.name]
    else:
        files = list(in_path.rglob(args.glob))
        outs  = [out_root / f.relative_to(in_path) for f in files]
    if not files:
        print("No input images found."); return
    for f, outf in tqdm(list(zip(files, outs)), total=len(files), desc="Processing"):
        ok, err = process_image(f, outf, args, yunet, veh_model, plate_model)
        if not ok:
            tqdm.write(f"[WARN] {f}: {err}")


if __name__ == "__main__":
    main()
