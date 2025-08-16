import argparse
import ssl
import urllib.request
from pathlib import Path
from typing import List, Tuple, Optional
import certifi
import cv2
import numpy as np
from tqdm import tqdm


def ensure_parent_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def clamp_geometry(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int,int,int,int]:
    x = max(0, min(W - 1, x)); y = max(0, min(H - 1, y))
    w = max(1, min(W - x, w)); h = max(1, min(H - y, h))
    return x, y, w, h

def filter_plate_roi_scores(boxes, img_shape, y_min_f, y_max_f):
    H = img_shape[0]
    y_min, y_max = H * y_min_f, H * y_max_f
    out = []
    for (x,y,w,h,s) in boxes:
        cy = y + h/2.0
        if y_min <= cy <= y_max:
            out.append((x,y,w,h,s))
    return out

def filter_by_geometry_scores(
    boxes: List[Tuple[int,int,int,int,float]],
    img_shape,
    min_size: int,
    ar_min: float,
    ar_max: float
) -> List[Tuple[int,int,int,int,float]]:
    """Keep boxes that satisfy size/aspect ratio; clamp to image; preserve score."""
    H, W = img_shape[:2]
    out: List[Tuple[int,int,int,int,float]] = []
    for (x, y, w, h, s) in boxes:
        if w < min_size or h < min_size:
            continue
        ar = w / max(1e-6, h)
        if ar < ar_min or ar > ar_max:
            continue
        x, y, w, h = clamp_geometry(x, y, w, h, W, H)
        out.append((x, y, w, h, s))
    return out


# Faces with Yunet
def _ensure_yunet_weights() -> str:
    weights_dir = Path("weights"); weights_dir.mkdir(parents=True, exist_ok=True)
    path = weights_dir / "face_detection_yunet_2023mar.onnx"
    if not path.exists():
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=ctx) as r, open(path, "wb") as f:
            f.write(r.read())
    return str(path)

def load_yunet(score_threshold: float = 0.6, nms_threshold: float = 0.3,
               top_k: int = 5000, model_size: int = 320):
    model_path = _ensure_yunet_weights()
    yunet = cv2.FaceDetectorYN_create(
        model_path, "", (model_size, model_size),
        score_threshold=score_threshold,
        nms_threshold=nms_threshold, top_k=top_k
    )
    return yunet

def detect_faces_yunet(frame_bgr: np.ndarray, yunet,
                       pad_x: float = 0.10, pad_y: float = 0.20) -> List[Tuple[int,int,int,int,float]]:
    H, W = frame_bgr.shape[:2]
    yunet.setInputSize((W, H))
    _, dets = yunet.detect(frame_bgr)
    boxes: List[Tuple[int,int,int,int,float]] = []
    if dets is not None:
        # det row: [x, y, w, h, score, ...landmarks]
        for d in dets:
            x, y, w, h = map(int, d[:4]); score = float(d[4])
            score = float(d[-1])
            dx, dy = int(pad_x * w), int(pad_y * h)
            x0 = max(0, x - dx); y0 = max(0, y - dy)
            x1 = min(W - 1, x + w + dx); y1 = min(H - 1, y + h + dy)
            if x1 > x0 and y1 > y0:
                boxes.append((x0, y0, x1 - x0, y1 - y0, score))
    return boxes


#Plates with Yolo
def load_plate_detector(model_arg: str):
    from ultralytics import YOLO
    m = YOLO(model_arg)
    _ = m.predict(source=np.zeros((10, 10, 3), dtype=np.uint8), conf=0.01, verbose=False)
    return m

def detect_plates(frame_bgr: np.ndarray, model,
                  conf_thres: float = 0.25,
                  iou_thres: float = 0.7,
                  imgsz: int = 640
                 ) -> List[Tuple[int,int,int,int,float]]:
    H, W = frame_bgr.shape[:2]
    boxes: List[Tuple[int,int,int,int,float]] = []
    results = model.predict(source=frame_bgr,
                            conf=conf_thres,
                            iou=iou_thres,
                            imgsz=imgsz,
                            verbose=False)
    for r in results:
        names = r.names
        for b in r.boxes:
            cls_id = int(b.cls[0])
            cls_name = names.get(cls_id, str(cls_id)).lower()
            if "plate" in cls_name or "license" in cls_name:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0])
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(W - 1, x2); y2 = min(H - 1, y2)
                if x2 > x1 and y2 > y1:
                    dx = int(0.03 * (x2 - x1)); dy = int(0.08 * (y2 - y1))
                    x0 = max(0, x1 - dx); y0 = max(0, y1 - dy)
                    x3 = min(W - 1, x2 + dx); y3 = min(H - 1, y2 + dy)
                    boxes.append((x0, y0, x3 - x0, y3 - y0, conf))
    return boxes



# plate color filter
def red_fraction(img: np.ndarray, box: Tuple[int,int,int,int,float]) -> float:
    x, y, w, h, _ = box
    roi = img[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 80, 50), (10, 255, 255))
    m2 = cv2.inRange(hsv, (160, 80, 50), (180, 255, 255))
    red = cv2.bitwise_or(m1, m2)
    return float(cv2.countNonZero(red)) / (roi.shape[0] * roi.shape[1])

def filter_red_dominant_scores(img: np.ndarray,
                               boxes: List[Tuple[int,int,int,int,float]],
                               thr: float) -> List[Tuple[int,int,int,int,float]]:
    return [b for b in boxes if red_fraction(img, b) <= thr]

def light_fraction(img: np.ndarray, box: Tuple[int,int,int,int,float],
                   s_max: int = 70, v_min: int = 150) -> float:
    x, y, w, h, _ = box
    roi = img[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s = hsv[:,:,1]; v = hsv[:,:,2]
    mask = (s <= s_max) & (v >= v_min)
    return float(np.count_nonzero(mask)) / (roi.shape[0] * roi.shape[1])

def filter_light_bg_scores(img: np.ndarray,
                           boxes: List[Tuple[int,int,int,int,float]],
                           thr: float, s_max: int = 70, v_min: int = 150
                          ) -> List[Tuple[int,int,int,int,float]]:
    return [b for b in boxes if light_fraction(img, b, s_max, v_min) >= thr]


# merge
def iou_xywh(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1 = max(ax, bx); y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw); y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    ua = aw * ah + bw * bh - inter
    return inter / max(1e-6, ua)

def merge_overlaps_scores(
    boxes: List[Tuple[int,int,int,int,float]],
    thr: float = 0.5
) -> List[Tuple[int,int,int,int,float]]:
    if not boxes:
        return boxes
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        out: List[Tuple[int,int,int,int,float]] = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]: continue
            ax, ay, aw, ah, ascore = boxes[i]
            merged = (ax, ay, aw, ah, ascore)
            for j in range(i + 1, len(boxes)):
                if used[j]: continue
                bx, by, bw, bh, bscore = boxes[j]
                if iou_xywh((merged[0], merged[1], merged[2], merged[3]), (bx, by, bw, bh)) >= thr:
                    x = min(merged[0], bx); y = min(merged[1], by)
                    r = max(merged[0] + merged[2], bx + bw)
                    btm = max(merged[1] + merged[3], by + bh)
                    merged = (x, y, r - x, btm - y, max(merged[4], bscore))
                    used[j] = True; changed = True
            used[i] = True; out.append(merged)
        boxes = out
    return boxes


# plate size cap
def filter_plate_maxsize_scores(
    boxes: List[Tuple[int,int,int,int,float]],
    img_shape, wf: float, hf: float, af: float
) -> List[Tuple[int,int,int,int,float]]:
    H, W = img_shape[:2]
    max_w, max_h, max_a = W * wf, H * hf, (W * H) * af
    out: List[Tuple[int,int,int,int,float]] = []
    for (x, y, w, h, s) in boxes:
        if w > max_w or h > max_h or (w * h) > max_a:
            continue
        out.append((x, y, w, h, s))
    return out


def gaussian_blur_roi(img: np.ndarray, x: int, y: int, w: int, h: int, k: Optional[int]):
    roi = img[y:y+h, x:x+w]
    if roi.size == 0:
        return
    if k is None:
        k = max(5, (min(w, h) // 6) | 1)
        if k % 2 == 0:
            k += 1
    img[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (k, k), 0)

def pixelate_roi(img: np.ndarray, x: int, y: int, w: int, h: int, pix: int):
    roi = img[y:y+h, x:x+w]
    if roi.size == 0:
        return
    ws = max(1, w // max(1, pix)); hs = max(1, h // max(1, pix))
    small = cv2.resize(roi, (ws, hs), interpolation=cv2.INTER_LINEAR)
    img[y:y+h, x:x+w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def blur_box(img: np.ndarray, box: Tuple[int,int,int,int], mode: str,
             kernel: Optional[int], pixel_size: int):
    x, y, w, h = box
    if mode == "pixelate":
        pixelate_roi(img, x, y, w, h, pixel_size)
    else:
        gaussian_blur_roi(img, x, y, w, h, kernel)

def draw_box_with_label(img: np.ndarray, box: Tuple[int,int,int,int], label: str,
                        conf: Optional[float] = None,
                        color: Tuple[int,int,int] = (0, 180, 255), thickness: int = 2):
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{label} {conf:.2f}" if conf is not None else label
    scale = 0.5; t_thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, t_thick)
    y_text = max(th + 4, y)
    cv2.rectangle(img, (x, y_text - th - 4), (x + tw + 4, y_text), color, -1)
    cv2.putText(img, text, (x + 2, y_text - 2), font, scale, (255, 255, 255), t_thick, cv2.LINE_AA)


# driver
def process_image(in_path: Path, out_path: Path, args, yunet=None, plate=None):
    img = cv2.imread(str(in_path))
    if img is None:
        return False, f"Cannot read {in_path}"

    faces: List[Tuple[int,int,int,int,float]] = []
    plates: List[Tuple[int,int,int,int,float]] = []

    if not args.no_faces:
        faces = detect_faces_yunet(img, yunet)
        faces = filter_by_geometry_scores(
            faces, img.shape, args.face_min_size, args.face_ar_min, args.face_ar_max
        )

    if not args.no_plates:
        plates = detect_plates(img, plate, args.plate_conf, args.plate_iou, args.plate_imgsz)
        plates = filter_by_geometry_scores(
            plates, img.shape, args.plate_min_size, args.plate_ar_min, args.plate_ar_max
        )
        plates = filter_plate_maxsize_scores(
            plates, img.shape, args.plate_max_width_frac, args.plate_max_height_frac, args.plate_max_area_frac
        )
        plates = filter_plate_roi_scores(plates, img.shape,
                                 args.plate_ymin_frac, args.plate_ymax_frac)

        if args.plate_reject_red:
            plates = filter_red_dominant_scores(img, plates, args.plate_red_frac)
        if args.plate_require_light_bg:
            plates = filter_light_bg_scores(img, plates, args.plate_light_frac,
                                            args.plate_light_smax, args.plate_light_vmin)
        if not args.no_plate_merge:
            plates = merge_overlaps_scores(plates, thr=args.plate_merge_iou)
            plates = filter_plate_maxsize_scores(
                plates, img.shape, args.plate_max_width_frac, args.plate_max_height_frac, args.plate_max_area_frac
            )
        if args.plate_min_width is not None and args.plate_min_width > 0:
            plates = [b for b in plates if b[2] >= args.plate_min_width]

    if args.mode == "boxes":
        for (x, y, w, h, s) in faces:
            draw_box_with_label(img, (x, y, w, h), "FACE", s)
        for (x, y, w, h, s) in plates:
            draw_box_with_label(img, (x, y, w, h), "PLATE", s)
    else:
        for (x, y, w, h, _) in faces + plates:
            blur_box(img, (x, y, w, h), args.blur,
                     args.kernel if (args.kernel and args.kernel % 2 == 1) else None,
                     args.pixel_size)

    ensure_parent_dir(out_path)
    ok = cv2.imwrite(str(out_path), img)

    if args.print_scores:
        def avg(xs): return sum(xs)/len(xs) if xs else 0.0
        face_scores = [s for *_, s in faces]
        plate_scores = [s for *_, s in plates]
        tqdm.write(
            f"{in_path.name}: faces={len(faces)} (avg {avg(face_scores):.3f}), "
            f"plates={len(plates)} (avg {avg(plate_scores):.3f})"
        )

    return ok, None if ok else f"Failed to write {out_path}"

def main():
    ap = argparse.ArgumentParser(description="Blur or box faces and license plates in images (minimal).")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--glob", default="*.png")
    ap.add_argument("--mode", choices=["blur", "boxes"], default="blur")
    ap.add_argument("--no-faces", action="store_true")
    ap.add_argument("--no-plates", action="store_true")

    # face params
    ap.add_argument("--face-min-size", type=int, default=16,
                    help="Minimum face box size in pixels (width/height).")
    ap.add_argument("--face-ar-min", type=float, default=0.50)
    ap.add_argument("--face-ar-max", type=float, default=2.40)
    ap.add_argument("--face-score-thres", type=float, default=0.6,
                    help="Lower to be looser (more faces, more false positives).")
    ap.add_argument("--face-nms-thres", type=float, default=0.3,
                    help="Higher to keep more overlapping face boxes.")
    ap.add_argument("--face-imgsz", type=int, default=320,
                    help="YuNet input size (square). Larger helps small/distant faces.")

    # blur params
    ap.add_argument("--blur", choices=["gaussian", "pixelate"], default="gaussian")
    ap.add_argument("--kernel", type=int, default=None)
    ap.add_argument("--pixel-size", type=int, default=12)

    # plate detector thresholds/filters
    ap.add_argument("--plate-ymin-frac", type=float, default=0.30)
    ap.add_argument("--plate-ymax-frac", type=float, default=0.95)
    ap.add_argument("--plate-model", type=str,
        default="https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1n.pt")
    ap.add_argument("--plate-conf", type=float, default=0.25,
                    help="Higher to be stricter (fewer plates, fewer false positives).")
    ap.add_argument("--plate-iou", type=float, default=0.7,
                    help="NMS IoU for YOLO; lower merges more duplicates (e.g., 0.5).")
    ap.add_argument("--plate-imgsz", type=int, default=640,
                    help="YOLO inference size (pixels). Larger helps small plates.")
    ap.add_argument("--plate-tta", action="store_true",
                    help="Enable YOLO test-time augmentation for higher recall.")
    ap.add_argument("--plate-min-size", type=int, default=10)
    ap.add_argument("--plate-min-width", type=int, default=40,
                    help="Minimum plate box width (pixels) after mapping to image; 0 to disable.")
    ap.add_argument("--plate-ar-min", type=float, default=0.90)
    ap.add_argument("--plate-ar-max", type=float, default=7.0)
    ap.add_argument("--plate-reject-red", action="store_true",
                    help="Drop plate boxes dominated by red pixels (taillights/signs).")
    ap.add_argument("--plate-red-frac", type=float, default=0.35,
                    help="Max red fraction (0..1) before rejecting a plate box.")
    ap.add_argument("--plate-require-light-bg", action="store_true",
                    help="Require a minimum fraction of light/low-saturation pixels in plate box.")
    ap.add_argument("--plate-light-frac", type=float, default=0.20)
    ap.add_argument("--plate-light-smax", type=int, default=70)
    ap.add_argument("--plate-light-vmin", type=int, default=130)
    ap.add_argument("--plate-merge-iou", type=float, default=0.5,
                    help="IoU threshold to merge overlapping plate boxes.")
    ap.add_argument("--no-plate-merge", action="store_true",
                    help="Disable post-merge of overlapping plate boxes.")
    # MAX SIZE caps
    ap.add_argument("--plate-max-width-frac",  type=float, default=0.60,
                    help="Drop plate boxes wider than this fraction of image width.")
    ap.add_argument("--plate-max-height-frac", type=float, default=0.28,
                    help="Drop plate boxes taller than this fraction of image height.")
    ap.add_argument("--plate-max-area-frac",   type=float, default=0.15,
                    help="Drop plate boxes larger than this fraction of image area.")
    ap.add_argument("--conf", type=float, default=None,
                    help="DEPRECATED alias for --plate-conf")
    ap.add_argument("--print-scores", action="store_true",
                    help="Print per-image detection counts and average confidences.")

    args = ap.parse_args()
    if args.conf is not None:
        args.plate_conf = args.conf
    yunet = None if args.no_faces else load_yunet(
        score_threshold=args.face_score_thres,
        nms_threshold=args.face_nms_thres,
        top_k=5000,
        model_size=args.face_imgsz
    )
    plate = None if args.no_plates else load_plate_detector(args.plate_model)

    in_path, out_root = Path(args.input), Path(args.output)
    if in_path.is_file():
        files = [in_path]
        outs = [out_root if out_root.suffix else out_root / in_path.name]
    else:
        files = list(in_path.rglob(args.glob))
        outs = [out_root / f.relative_to(in_path) for f in files]
    if not files:
        print("No input images found."); return

    for f, out_f in tqdm(list(zip(files, outs)), total=len(files), desc="Processing"):
        ok, err = process_image(f, out_f, args, yunet, plate)
        if not ok:
            tqdm.write(f"[WARN] {f}: {err}")

if __name__ == "__main__":
    main()
