import argparse
import ssl
import urllib.request
from pathlib import Path
from typing import List, Tuple
import certifi
import cv2
import numpy as np
from tqdm import tqdm


# ---------- small utils ----------
def ensure_parent_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def filter_by_geometry(
    boxes: List[Tuple[int,int,int,int]],
    img_shape,
    min_size: int,
    ar_min: float,
    ar_max: float
) -> List[Tuple[int,int,int,int]]:
    H, W = img_shape[:2]
    out = []
    for (x,y,w,h) in boxes:
        if w < min_size or h < min_size:
            continue
        ar = w / max(1e-6, h)
        if ar < ar_min or ar > ar_max:
            continue
        x = max(0, min(W-1, x)); y = max(0, min(H-1, y))
        w = max(1, min(W - x, w)); h = max(1, min(H - y, h))
        out.append((x,y,w,h))
    return out


# ---------- faces: YuNet ----------
def _ensure_yunet_weights() -> str:
    weights_dir = Path("weights"); weights_dir.mkdir(parents=True, exist_ok=True)
    path = weights_dir / "face_detection_yunet_2023mar.onnx"
    if not path.exists():
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=ctx) as r, open(path, "wb") as f:
            f.write(r.read())
    return str(path)

def load_yunet(score_threshold: float = 0.6, nms_threshold: float = 0.3, top_k: int = 5000):
    model_path = _ensure_yunet_weights()
    yunet = cv2.FaceDetectorYN_create(model_path, "", (320, 320),
                                      score_threshold=score_threshold,
                                      nms_threshold=nms_threshold, top_k=top_k)
    return yunet

def detect_faces_yunet(frame_bgr: np.ndarray, yunet,
                       pad_x=0.10, pad_y=0.20) -> List[Tuple[int,int,int,int]]:
    H, W = frame_bgr.shape[:2]
    yunet.setInputSize((W, H))
    _, dets = yunet.detect(frame_bgr)
    boxes: List[Tuple[int,int,int,int]] = []
    if dets is not None:
        for d in dets:  # [x, y, w, h, score, ...landmarks]
            x, y, w, h = map(int, d[:4])
            dx, dy = int(pad_x * w), int(pad_y * h)
            x0 = max(0, x - dx); y0 = max(0, y - dy)
            x1 = min(W - 1, x + w + dx); y1 = min(H - 1, y + h + dy)
            if x1 > x0 and y1 > y0:
                boxes.append((x0, y0, x1 - x0, y1 - y0))
    return boxes


# ---------- plates: YOLOv8/11 ----------
def load_plate_detector(model_arg: str):
    from ultralytics import YOLO
    m = YOLO(model_arg) 
    _ = m.predict(source=np.zeros((10, 10, 3), dtype=np.uint8), conf=0.01, verbose=False)  
    return m

def detect_plates(frame_bgr: np.ndarray, model, conf_thres: float = 0.25) -> List[Tuple[int,int,int,int]]:
    H, W = frame_bgr.shape[:2]
    boxes: List[Tuple[int,int,int,int]] = []
    results = model.predict(source=frame_bgr, conf=conf_thres, verbose=False)
    for r in results:
        names = r.names
        for b in r.boxes:
            cls_id = int(b.cls[0])
            cls_name = names.get(cls_id, str(cls_id)).lower()
            if "plate" in cls_name or "license" in cls_name:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(W - 1, x2); y2 = min(H - 1, y2)
                if x2 > x1 and y2 > y1:
                    dx = int(0.05 * (x2 - x1)); dy = int(0.10 * (y2 - y1))
                    x0 = max(0, x1 - dx); y0 = max(0, y1 - dy)
                    x3 = min(W - 1, x2 + dx); y3 = min(H - 1, y2 + dy)
                    boxes.append((x0, y0, x3 - x0, y3 - y0))
    return boxes


# ---------- render ----------
def gaussian_blur_roi(img: np.ndarray, x: int, y: int, w: int, h: int, k: int | None):
    roi = img[y:y+h, x:x+w]
    if roi.size == 0:
        return
    if k is None:
        k = max(5, (min(w, h) // 6) | 1)
        if k % 2 == 0: k += 1
    img[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (k, k), 0)

def pixelate_roi(img: np.ndarray, x: int, y: int, w: int, h: int, pix: int):
    roi = img[y:y+h, x:x+w]
    if roi.size == 0: return
    ws = max(1, w // max(1, pix)); hs = max(1, h // max(1, pix))
    small = cv2.resize(roi, (ws, hs), interpolation=cv2.INTER_LINEAR)
    img[y:y+h, x:x+w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def blur_box(img: np.ndarray, box: Tuple[int,int,int,int], mode: str, kernel: int | None, pixel_size: int):
    x,y,w,h = box
    if mode == "pixelate": pixelate_roi(img, x, y, w, h, pixel_size)
    else: gaussian_blur_roi(img, x, y, w, h, kernel)

def draw_box_with_label(img: np.ndarray, box: Tuple[int,int,int,int], label: str,
                        color: Tuple[int,int,int]=(0,180,255), thickness: int=2):
    x,y,w,h = box
    cv2.rectangle(img, (x,y), (x+w,y+h), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5; t_thick = 1
    (tw,th), _ = cv2.getTextSize(label, font, scale, t_thick)
    y_text = max(th+4, y)
    cv2.rectangle(img, (x, y_text-th-4), (x+tw+4, y_text), color, -1)
    cv2.putText(img, label, (x+2, y_text-2), font, scale, (255,255,255), t_thick, cv2.LINE_AA)


# ---------- driver ----------
def process_image(in_path: Path, out_path: Path, args, yunet=None, plate=None):
    img = cv2.imread(str(in_path))
    if img is None:
        return False, f"Cannot read {in_path}"

    faces: List[Tuple[int,int,int,int]] = []
    plates: List[Tuple[int,int,int,int]] = []

    if not args.no_faces:
        faces = detect_faces_yunet(img, yunet)
        faces = filter_by_geometry(faces, img.shape, args.face_min_size, args.face_ar_min, args.face_ar_max)

    if not args.no_plates:
        plates = detect_plates(img, plate, args.conf)

    if args.mode == "boxes":
        for b in faces:  draw_box_with_label(img, b, "FACE")
        for b in plates: draw_box_with_label(img, b, "PLATE")
    else:
        for b in faces + plates:
            blur_box(img, b, args.blur, args.kernel if (args.kernel and args.kernel % 2 == 1) else None, args.pixel_size)

    ensure_parent_dir(out_path)
    ok = cv2.imwrite(str(out_path), img)
    return ok, None if ok else f"Failed to write {out_path}"

def main():
    ap = argparse.ArgumentParser(description="Blur or box faces and license plates in images (minimal).")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--glob", default="*.png")
    ap.add_argument("--mode", choices=["blur","boxes"], default="blur")
    ap.add_argument("--no-faces", action="store_true")
    ap.add_argument("--no-plates", action="store_true")

    # face params
    ap.add_argument("--face-min-size", type=int, default=14)
    ap.add_argument("--face-ar-min", type=float, default=0.45)
    ap.add_argument("--face-ar-max", type=float, default=2.2)

    # blur params
    ap.add_argument("--blur", choices=["gaussian","pixelate"], default="gaussian")
    ap.add_argument("--kernel", type=int, default=None)
    ap.add_argument("--pixel-size", type=int, default=12)

    # plate detector
    ap.add_argument("--plate-model", type=str,
        default="https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1n.pt")
    ap.add_argument("--conf", type=float, default=0.25)

    args = ap.parse_args()

    # load detectors
    yunet = None if args.no_faces else load_yunet(score_threshold=0.6, nms_threshold=0.3, top_k=5000)
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
        if not ok: tqdm.write(f"[WARN] {f}: {err}")

if __name__ == "__main__":
    main()
