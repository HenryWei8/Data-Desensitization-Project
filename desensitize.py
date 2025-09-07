import cv2, numpy as np, re
from ultralytics import YOLO
from pathlib import Path

HP = dict(
    INPUT="data/imgs",
    OUTPUT="mosaic",
    GLOB=r"*.[Pp][Nn][Gg]",
    MODE="blur",

    PEOPLE_MODEL="yolov8x.pt",
    PEOPLE_CONF=0.15, PEOPLE_IOU=0.60, PEOPLE_IMGSZ=1792,
    NO_PEOPLE=False,

    FACE_MODEL="weights/yolov8x-face.pt",
    FACE_HEAD_CONF=0.01, FACE_HEAD_IOU=0.60, FACE_HEAD_IMGSZ=1024,
    HEAD_FRAC=0.20, HEAD_WIDTH_FRAC=0.55, HEAD_TOP_BIAS=0.02, HEAD_PAD_FRAC=0.05,
    NO_FACES=False,

    VEHICLE_MODEL="yolov8n.pt",
    VEHICLE_CONF=0.15, VEHICLE_IOU=0.70, VEHICLE_IMGSZ=2304,
    VEHICLE_PAD_FRAC=0.12, VEHICLE_AGNOSTIC_NMS=False,
    NO_VEHICLES=False,

    PLATE_MODEL_MAIN="weights/plate-v1n.pt",
    PLATE_ACCEPT_GLOBAL=True,
    PLATE_PROBE_CONF=0.001, PLATE_GLOBAL_CONF=0.20,
    PLATE_IOU=0.60, PLATE_IMGSZ=3584,
    PLATE_MAX_W_FRAC=0.30, PLATE_MAX_H_FRAC=0.20,
    PLATE_MAX_AREA_WRT_VEH=0.60,
    PLATE_MIN_W_PX=10, PLATE_MIN_H_PX=6,
    PLATE_MIN_W_FRAC=0.002, PLATE_MIN_H_FRAC=0.0015,
    PLATE_MIN_CONF=0.015,
    REQUIRE_VEHICLE_OVERLAP=True,
    PLATE_VEHICLE_IOU=0.01,
    PLATE_MERGE_IOU=0.50,
    ORPHAN_PLATE_KEEP_CONF=0.40,
    MIN_PLATE_COVERAGE=0.45,
    NO_PLATES=False,

    PLATE_MODEL_EU="weights/plate-eu-v8.pt",
    USE_EU_ENSEMBLE=True,

    RELAX_TEMPORAL=True,
    RELAX_CONF_MULT=0.5,
    RELAX_CONF_FLOOR=0.006,

    BLUR_KIND="pixelate",
    KERNEL=None, PIXEL_SIZE=8,
    BLUR_PAD_FRAC_PLATE=0.10,
    VIZ_ALL_PLATES=False,
    DEBUG=False,
)

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

def _box_iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1 = max(ax, bx); y1 = max(ay, by)
    x2 = min(ax+aw, bx+bw); y2 = min(ay+ah, by+bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = aw*ah + bw*bh - inter
    return inter/union if union>0 else 0.0

def _coverage(a, b):
    ax,ay,aw,ah = a; bx,by,bw,bh = b
    x1 = max(ax, bx); y1 = max(ay, by)
    x2 = min(ax+aw, bx+bw); y2 = min(ay+ah, by+bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    return inter / float(max(1, aw*ah))

def _pad_frac(box, frac, W, H):
    if frac <= 0: return box
    x,y,w,h = box
    px = int(round(frac*w)); py = int(round(frac*h))
    x -= px; y -= py; w += 2*px; h += 2*py
    return clamp_box(x,y,w,h,W,H)

def draw_box(img, box, label, conf=None, color=(0,255,0), thick=2):
    x,y,w,h = box
    cv2.rectangle(img, (x,y), (x+w,y+h), color, thick)
    text = f"{label} {conf:.2f}" if (conf is not None) else label
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y2 = max(th + 4, y)
    cv2.rectangle(img, (x, y2 - th - 4), (x + tw + 6, y2), color, -1)
    cv2.putText(img, text, (x + 3, y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

def _natural_composite(roi, blur, feather=10, round_frac=0.18):
    h, w = roi.shape[:2]
    r = max(1, min(int(round(round_frac * min(w, h))), min(w, h) // 2))
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
    cv2.circle(mask, (r, r), r, 255, -1); cv2.circle(mask, (w - r, r), r, 255, -1)
    cv2.circle(mask, (r, h - r), r, 255, -1); cv2.circle(mask, (w - r, h - r), r, 255, -1)
    if feather > 0:
        f = int(feather * 2 + 1)
        mask = cv2.GaussianBlur(mask, (f, f), 0)
    a = (mask.astype(np.float32) / 255.0)[..., None]
    return (blur.astype(np.float32) * a + roi.astype(np.float32) * (1 - a)).astype(np.uint8)

def gaussian_blur_roi(img, x, y, w, h, k=None):
    H, W = img.shape[:2]
    x,y,w,h = clamp_box(x,y,w,h,W,H)
    roi = img[y:y+h, x:x+w]
    if roi.size == 0: return
    if k is None: k = max(5, (min(w,h)//6) | 1)
    if k % 2 == 0: k += 1
    blur = cv2.GaussianBlur(roi, (k,k), 0)
    img[y:y+h, x:x+w] = _natural_composite(roi, blur, feather=10, round_frac=0.18)

def pixelate_roi(img, x, y, w, h, pix=12):
    H, W = img.shape[:2]
    x,y,w,h = clamp_box(x,y,w,h,W,H)
    roi = img[y:y+h, x:x+w]
    if roi.size == 0: return
    rw, rh = roi.shape[1], roi.shape[0]
    ws = max(1, rw // max(1, pix)); hs = max(1, rh // max(1, pix))
    small = cv2.resize(roi, (ws, hs), interpolation=cv2.INTER_LINEAR)
    blur = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    img[y:y+h, x:x+w] = _natural_composite(roi, blur, feather=10, round_frac=0.18)

def load_yolo(model_path: str):
    m = YOLO(model_path)
    _ = m.predict(source=np.zeros((10,10,3), dtype=np.uint8), conf=0.01, verbose=False)
    return m

def yolo_detect(img_bgr, model, conf=0.25, iou=0.5, imgsz=640, name_allow=None, agnostic=True):
    H, W = img_bgr.shape[:2]
    out = []
    for r in model.predict(source=img_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False, agnostic_nms=agnostic):
        names = r.names
        for b in r.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            c = float(b.conf[0]); cls = int(b.cls[0])
            name = names.get(cls, str(cls)).lower()
            if (name_allow is None) or (name in name_allow):
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(W - 1, x2); y2 = min(H - 1, y2)
                if x2 > x1 and y2 > y1:
                    out.append((x1,y1,x2-x1,y2-y1,c,name))
    return out

def detect_people(img_bgr, people_model):
    if HP["NO_PEOPLE"] or people_model is None: return []
    return yolo_detect(img_bgr, people_model, HP["PEOPLE_CONF"], HP["PEOPLE_IOU"], HP["PEOPLE_IMGSZ"], name_allow=["person"])

def faces_from_people_heads(img_bgr, face_model, people):
    if HP["NO_FACES"] or not people or face_model is None: return []
    H, W = img_bgr.shape[:2]
    out = []
    for (px, py, pw, ph, _, _) in people:
        hh = max(14, min(int(round(ph * HP["HEAD_FRAC"])), 180))
        ww_cap = int(round(hh * 1.1))
        ww = max(8, min(int(round(pw * HP["HEAD_WIDTH_FRAC"])), ww_cap))
        cx = px + pw * 0.5
        x_fb = int(round(cx - ww * 0.5))
        y_fb = int(round(py + ph * HP["HEAD_TOP_BIAS"]))
        fx, fy, fw, fh = clamp_box(x_fb, y_fb, ww, hh, W, H)
        sx = max(0, fx - int(HP["HEAD_PAD_FRAC"] * fw))
        sy = max(0, fy - int(HP["HEAD_PAD_FRAC"] * fh))
        sw = min(W - sx, fw + int(2 * HP["HEAD_PAD_FRAC"] * fw))
        sh = min(H - sy, fh + int(2 * HP["HEAD_PAD_FRAC"] * fh))
        crop = img_bgr[sy:sy+sh, sx:sx+sw]
        best = (fx, fy, fw, fh, None); best_s = -1.0
        res = face_model.predict(source=crop, conf=HP["FACE_HEAD_CONF"], iou=HP["FACE_HEAD_IOU"], imgsz=HP["FACE_HEAD_IMGSZ"], verbose=False, agnostic_nms=True)
        for r in res:
            names = r.names
            for b in r.boxes:
                name = names.get(int(b.cls[0]), "?").lower()
                if "face" not in name: continue
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist()); s = float(b.conf[0])
                gx1 = np.clip(sx + x1, 0, W - 1); gy1 = np.clip(sy + y1, 0, H - 1)
                gx2 = np.clip(sx + x2, 0, W - 1); gy2 = np.clip(sy + y2, 0, H - 1)
                if gx2 <= gx1 or gy2 <= gy1: continue
                gw, gh = gx2 - gx1, gy2 - gy1
                fcx = gx1 + gw * 0.5; fcy = gy1 + gh * 0.5
                if not (px <= fcx <= px + pw and py <= fcy <= py + ph * 0.5): continue
                ar = gw / max(1.0, gh)
                if not (0.7 <= ar <= 1.45): continue
                if s > best_s: best_s = s; best = (int(gx1), int(gy1), int(gw), int(gh), s)
        out.append(best)
    return out

def _detect_plates_with_model(img_bgr, plate_model, conf, iou, imgsz):
    if plate_model is None: return []
    H, W = img_bgr.shape[:2]
    out = []
    res = plate_model.predict(source=img_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False, agnostic_nms=True)
    for r in res:
        names = r.names
        for b in r.boxes:
            name = names.get(int(b.cls[0]), "?").lower()
            if ("plate" not in name) and ("license" not in name): continue
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist()); s = float(b.conf[0])
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(W - 1, x2); y2 = min(H - 1, y2)
            if x2 > x1 and y2 > y1:
                out.append((x1, y1, x2 - x1, y2 - y1, s, "global"))
    return out

def detect_plates_global(img_bgr, models):
    out = []
    for m in models:
        out.extend(_detect_plates_with_model(img_bgr, m, HP["PLATE_GLOBAL_CONF"], HP["PLATE_IOU"], HP["PLATE_IMGSZ"]))
    return out

def collect_plate_candidates(img_bgr, models, vehicles):
    H, W = img_bgr.shape[:2]
    cands = []
    if HP["PLATE_ACCEPT_GLOBAL"]:
        cands.extend(detect_plates_global(img_bgr, models))
    for vi, (vx, vy, vw, vh, _, vname) in enumerate(vehicles):
        vx,vy,vw,vh = clamp_box(vx,vy,vw,vh,W,H)
        crop = img_bgr[vy:vy+vh, vx:vx+vw]
        for m in models:
            if m is None: continue
            res = m.predict(source=crop, conf=HP["PLATE_PROBE_CONF"], iou=HP["PLATE_IOU"], imgsz=HP["PLATE_IMGSZ"], verbose=False, agnostic_nms=True)
            for r in res:
                names = r.names
                for b in r.boxes:
                    name = names.get(int(b.cls[0]), "?").lower()
                    if ("plate" not in name) and ("license" not in name): continue
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist()); s = float(b.conf[0])
                    x1 += vx; y1 += vy; x2 += vx; y2 += vy
                    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W - 1, x2); y2 = min(H - 1, y2)
                    if x2 <= x1 or y2 <= y1: continue
                    cands.append((x1, y1, x2 - x1, y2 - y1, s, f"veh{vi}"))
    return cands

def iou_xywh(a, b): return _box_iou_xywh(a, b)

def filter_and_dedup(cands, vehicles, img_wh):
    W, H = img_wh
    kept = []
    for (x, y, w, h, s, src) in cands:
        if s < HP["PLATE_MIN_CONF"]: continue
        mw = max(HP["PLATE_MIN_W_PX"], int(HP["PLATE_MIN_W_FRAC"] * W))
        mh = max(HP["PLATE_MIN_H_PX"], int(HP["PLATE_MIN_H_FRAC"] * H))
        if w < mw or h < mh: continue
        if w > int(HP["PLATE_MAX_W_FRAC"] * W) or h > int(HP["PLATE_MAX_H_FRAC"] * H): continue
        ar = w / max(1.0, h)
        if not (1.1 <= ar <= 10): continue
        plate = (x, y, w, h)
        associated = False
        if HP["REQUIRE_VEHICLE_OVERLAP"] and vehicles:
            best_cov, best_v = 0.0, None
            for (vx, vy, vw, vh, _, _) in vehicles:
                if HP["VEHICLE_PAD_FRAC"] > 0:
                    px = max(0, vx - int(HP["VEHICLE_PAD_FRAC"] * vw))
                    py = max(0, vy - int(HP["VEHICLE_PAD_FRAC"] * vh))
                    pw = min(W - px, vw + int(2 * HP["VEHICLE_PAD_FRAC"] * vw))
                    ph = min(H - py, vh + int(2 * HP["VEHICLE_PAD_FRAC"] * vh))
                    vbox = (px, py, pw, ph)
                else:
                    vbox = (vx, vy, vw, vh)
                cov = _coverage(plate, vbox)
                if cov > best_cov: best_cov, best_v = cov, vbox
            if best_cov >= HP["MIN_PLATE_COVERAGE"]:
                vx, vy, vw, vh = best_v
                if (w*h) <= HP["PLATE_MAX_AREA_WRT_VEH"] * (vw*vh):
                    associated = True
        if associated or (s >= HP["ORPHAN_PLATE_KEEP_CONF"]):
            kept.append((x, y, w, h, s, src))
    kept.sort(key=lambda t: t[4], reverse=True)
    final = []
    for cand in kept:
        if all(iou_xywh((cand[0], cand[1], cand[2], cand[3]), (b[0], b[1], b[2], b[3])) < HP["PLATE_MERGE_IOU"] for b in final):
            final.append(cand)
    return final

PLATE_TRACKS = {}
_NEXT_TRACK_ID = 1

def _relaxed_one_plate_per_vehicle(vehicles, plates, img_wh):
    global PLATE_TRACKS, _NEXT_TRACK_ID
    W, H = img_wh
    used_tracks = set()
    out = []
    for i,(vx,vy,vw,vh,_,_) in enumerate(vehicles):
        vbox = (vx,vy,vw,vh)
        tid, best = None, 0.0
        for k,t in PLATE_TRACKS.items():
            iou = _box_iou_xywh(t["veh"], vbox)
            if iou >= 0.35 and iou > best:
                best, tid = iou, k
        if tid is None:
            tid = _NEXT_TRACK_ID; _NEXT_TRACK_ID += 1
            PLATE_TRACKS[tid] = {"veh": vbox, "had_plate": False}
        used_tracks.add(tid)

        thresh = HP["PLATE_MIN_CONF"]
        if HP["RELAX_TEMPORAL"] and PLATE_TRACKS[tid]["had_plate"]:
            thr_relax = max(HP["RELAX_CONF_FLOOR"], HP["PLATE_MIN_CONF"] * HP["RELAX_CONF_MULT"])
            thresh = min(thresh, thr_relax)

        best_det = None; best_key = (-1e9,)
        for (x,y,w,h,s,src) in plates:
            cx = x + w*0.5; cy = y + h*0.5
            if not (vx <= cx <= vx+vw and vy <= cy <= vy+vh): continue
            if s < thresh: continue
            key = (s,)
            if key > best_key:
                best_key, best_det = key, (x,y,w,h,s,src)
        if best_det:
            out.append(best_det)
            PLATE_TRACKS[tid]["had_plate"] = True
        else:
            PLATE_TRACKS[tid]["had_plate"] = False
        PLATE_TRACKS[tid]["veh"] = vbox

    for k in list(PLATE_TRACKS.keys()):
        if k not in used_tracks:
            del PLATE_TRACKS[k]
    return out if out else plates

def process_image(path_in, path_out, people_m, face_m, veh_m, plate_models):
    img = cv2.imread(str(path_in))
    if img is None: return False, f"Cannot read {path_in}"
    H, W = img.shape[:2]
    people = detect_people(img, people_m)
    faces  = faces_from_people_heads(img, face_m, people)
    vehicles = [] if HP["NO_VEHICLES"] else yolo_detect(
        img, veh_m, conf=HP["VEHICLE_CONF"], iou=HP["VEHICLE_IOU"], imgsz=HP["VEHICLE_IMGSZ"],
        name_allow=["car","truck","bus","motorcycle","motorbike"], agnostic=HP["VEHICLE_AGNOSTIC_NMS"]
    )
    cands = [] if HP["NO_PLATES"] else collect_plate_candidates(img, plate_models, vehicles)
    plates = [] if HP["NO_PLATES"] else filter_and_dedup(cands, vehicles, (W, H))
    if vehicles and not HP["NO_PLATES"]:
        plates = _relaxed_one_plate_per_vehicle(vehicles, plates, (W, H))
    if HP["MODE"] == "boxes":
        for (x,y,w,h,s,_) in vehicles: draw_box(img, (x,y,w,h), "VEHICLE", s, (0,255,0), 2)
        for (x,y,w,h,s,_) in people:   draw_box(img, (x,y,w,h), "PERSON", s, (0,200,255), 2)
        for (x,y,w,h,s) in faces:      draw_box(img, (x,y,w,h), "FACE", s, (255,100,0), 2)
        for (x,y,w,h,s,src) in plates: draw_box(img, (x,y,w,h), "PLATE", s if s>=0.05 else None, (0,180,255), 2)
        if HP["VIZ_ALL_PLATES"]:
            for (x,y,w,h,s,src) in cands: draw_box(img, (x,y,w,h), f"ALL {s:.2f}", s, (255,0,255), 2)
    else:
        rois = []
        if not HP["NO_PLATES"]:
            for (x,y,w,h,_,_) in plates:
                x,y,w,h = _pad_frac((x,y,w,h), HP["BLUR_PAD_FRAC_PLATE"], W, H)
                rois.append((x,y,w,h))
        if not HP["NO_FACES"]:
            rois += [(x,y,w,h) for (x,y,w,h,_) in faces]
        k = HP["KERNEL"] if (HP["KERNEL"] is None or HP["KERNEL"] % 2 == 1) else HP["KERNEL"] + 1
        for (x,y,w,h) in rois:
            if HP["BLUR_KIND"] == "pixelate": pixelate_roi(img, x, y, w, h, HP["PIXEL_SIZE"])
            else: gaussian_blur_roi(img, x, y, w, h, k)
    ensure_parent_dir(path_out)
    ok = cv2.imwrite(str(path_out), img)
    return ok, None if ok else f"Failed to write {path_out}"

def main():
    global PLATE_TRACKS, _NEXT_TRACK_ID
    PLATE_TRACKS.clear(); _NEXT_TRACK_ID = 1
    people_m = None if HP["NO_PEOPLE"] else load_yolo(HP["PEOPLE_MODEL"])
    face_m   = None if HP["NO_FACES"]  else load_yolo(HP["FACE_MODEL"])
    veh_m    = None if HP["NO_VEHICLES"] else load_yolo(HP["VEHICLE_MODEL"])
    plate_models = []
    if Path(HP["PLATE_MODEL_MAIN"]).exists(): plate_models.append(load_yolo(HP["PLATE_MODEL_MAIN"]))
    else: plate_models.append(None)
    if HP["USE_EU_ENSEMBLE"] and Path(HP["PLATE_MODEL_EU"]).exists():
        plate_models.append(load_yolo(HP["PLATE_MODEL_EU"]))
    in_path, out_root = Path(HP["INPUT"]), Path(HP["OUTPUT"])
    if in_path.is_file():
        files = [in_path]; outs = [out_root if out_root.suffix else out_root / in_path.name]
    else:
        def _nat_key(p): return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(p))]
        files = sorted(in_path.rglob(HP["GLOB"]), key=_nat_key)
        outs  = [out_root / f.relative_to(in_path) for f in files]
    if not files: print("No input images found."); return
    for f, outf in zip(files, outs):
        ok, err = process_image(f, outf, people_m, face_m, veh_m, plate_models)
        if not ok: print(f"[WARN] {f}: {err}")

if __name__ == "__main__":
    main()


