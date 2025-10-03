# backend/main.py
# -*- coding: utf-8 -*-
import os
import shutil
import json
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import threading

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import aiofiles

# image libs
import cv2
from pyzbar.pyzbar import decode
from PIL import Image, ImageDraw, ImageFont

# ----------------- Config -----------------
APP_TITLE = "TOOL DANH SO XUONG - API"
IMG_EXTS = (".jpg", ".jpeg", ".png")
ORDERED_FOLDERS = [
    ("don nhieu items", "nhieu item"),
    ("kidshirt", "kidshirt"),
    ("1item", "1item"),
    ("don hieu ung mau toi", "hieuungmautoi"),
    ("don khach tu sua", "khachtusua"),
]
FONT_PATH = "arialbd.ttf"
FONT_SIZE = 240
TEXT_COLOR = (255, 0, 0, 255)
MARGIN_EXPAND1 = 500
MARGIN_EXPAND2 = 150
TEXT_MARGIN = 40
MAX_HEIGHT_PX = 75000

# workspace root (relative to this file)
BASE = Path(__file__).parent.resolve()
WORKSPACE = BASE / "workspace"
for d in ["tudong_input", "completed", "error_images", "output_images", "input_images", "uploads"]:
    (WORKSPACE / d).mkdir(parents=True, exist_ok=True)

STATE_FILE = WORKSPACE / "state.json"
REPORT_DIR = WORKSPACE

# ----------------- helpers -----------------
def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

def load_state():
    default = {"global_index": 0, "folder_index": {}, "processed_files": []}
    if STATE_FILE.exists():
        try:
            s = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            default.update(s)
        except Exception:
            pass
    for _f, label in ORDERED_FOLDERS:
        default["folder_index"].setdefault(label, 0)
    return default

def unique_path(target: Path) -> Path:
    if not target.exists():
        return target
    stem, ext = target.stem, target.suffix
    i = 1
    while True:
        cand = target.with_name(f"{stem}-{i}{ext}")
        if not cand.exists():
            return cand
        i += 1

# ----------------- QRCodeDetector -----------------
class QRCodeDetector:
    def count_qr_codes(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            return 0, "Unknown"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raw_qr_codes = decode(gray)

        def is_similar_rect(r1, r2, tolerance=10):
            return all(abs(a - b) <= tolerance for a, b in zip(r1, r2))

        seen_rects = []
        filtered_qr_codes = []
        for qr in raw_qr_codes:
            rect = qr.rect
            if hasattr(rect, "_fields"):
                key = (rect.left, rect.top, rect.width, rect.height)
            else:
                key = tuple(rect)
            if not any(is_similar_rect(key, seen_key) for seen_key in seen_rects):
                seen_rects.append(key)
                filtered_qr_codes.append(qr)

        count = len(filtered_qr_codes)
        layout = self._determine_layout(filtered_qr_codes, img.shape)
        return count, layout

    def _determine_layout(self, qr_codes, image_shape):
        if not qr_codes:
            return "Unknown"
        try:
            y_coords = [qr.rect.top for qr in qr_codes]
            x_coords = [qr.rect.left for qr in qr_codes]
        except Exception:
            coords = []
            for qr in qr_codes:
                try:
                    x, y, w, h = qr.rect
                    coords.append((x, y))
                except Exception:
                    pass
            if not coords:
                return "Unknown"
            x_coords = [c[0] for c in coords]
            y_coords = [c[1] for c in coords]
        y_spread = max(y_coords) - min(y_coords) if y_coords else 0
        x_spread = max(x_coords) - min(x_coords) if x_coords else 0
        if y_spread > x_spread:
            return "Vertical"
        elif x_spread > y_spread:
            return "Horizontal"
        return "Mixed"

qr_detector = QRCodeDetector()

# ----------------- image processing (port add_text_with_canvas_expansion) -----------------
def add_text_with_canvas_expansion_local(image_path: Path, output_path: Path, text: str,
                                        no_limit=False, rotate_back=True, check_rotate=True):
    Image.MAX_IMAGE_PIXELS = None
    try:
        original = Image.open(str(image_path))
        original_format = original.format or "PNG"
        dpi = original.info.get("dpi", (300, 300))
        mode = original.mode
        w, h = original.size
        rotated = False
        if check_rotate and w > h:
            original = original.rotate(90, expand=True)
            w, h = original.size
            rotated = True
        bg = (255, 255, 255, 0) if mode == "RGBA" else (255, 255, 255)
        # load font
        try:
            font = ImageFont.truetype(str(Path(FONT_PATH)), FONT_SIZE)
        except Exception:
            try:
                font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
            except Exception:
                font = ImageFont.load_default()
        bbox = ImageDraw.Draw(original).textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        new_h1 = h + 2 * MARGIN_EXPAND1
        if not no_limit and new_h1 > MAX_HEIGHT_PX:
            return False, f"Ảnh quá cao: {new_h1}px > {MAX_HEIGHT_PX}px"
        step1_img = Image.new("RGBA" if mode == "RGBA" else "RGB", (w, new_h1), bg)
        step1_img.paste(original, (0, MARGIN_EXPAND1))
        d1 = ImageDraw.Draw(step1_img)
        top_y = TEXT_MARGIN
        d1.text(((w - tw) // 2, top_y), text, fill=TEXT_COLOR, font=font)
        bottom_y = new_h1 - th - TEXT_MARGIN
        d1.text(((w - tw) // 2, bottom_y), text, fill=TEXT_COLOR, font=font)
        new_h2 = new_h1 + 2 * MARGIN_EXPAND2
        final_img = Image.new("RGBA" if mode == "RGBA" else "RGB", (w, new_h2), bg)
        final_img.paste(step1_img, (0, MARGIN_EXPAND2))
        if rotated and rotate_back:
            final_img = final_img.rotate(-90, expand=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_img.save(str(output_path), format=original_format, dpi=dpi)
        return True, None
    except Exception as e:
        return False, str(e)

# ----------------- processing pipeline -----------------
def today_dd_mm():
    return datetime.now().strftime("%d-%m")

def run_tudong_on_folder_sync(input_dir: Path, output_dir: Path, error_dir: Path,
                              workers: int, no_limit: bool, rotate_back: bool, check_rotate: bool,
                              progress_cb=None, log_cb=None):
    files = [p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()]
    total = len(files)
    if total == 0:
        return
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for f in files:
            out_path = output_dir / f.name
            task = (f, out_path, f.stem, no_limit, rotate_back, check_rotate)
            futures[ex.submit(add_text_with_canvas_expansion_local, *task)] = f
        for fut in as_completed(futures):
            ok, msg = fut.result()
            fname = futures[fut].name
            completed += 1
            if progress_cb:
                progress_cb(completed, total)
            if ok:
                if log_cb:
                    log_cb(f"✔ Xong: {fname}")
            else:
                if log_cb:
                    log_cb(f"✘ Lỗi: {fname} -> {msg}")
                try:
                    shutil.move(str(futures[fut]), str(error_dir / fname))
                except Exception:
                    pass

# ----------------- WebSocket manager -----------------
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            self.active_connections.remove(ws)

    async def send_json(self, data: dict):
        to_remove = []
        for ws in self.active_connections:
            try:
                await ws.send_json(data)
            except Exception:
                to_remove.append(ws)
        for r in to_remove:
            self.disconnect(r)

manager = ConnectionManager()

# ----------------- FastAPI app -----------------
app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/state")
async def api_state():
    return load_state()

@app.post("/api/upload")
async def api_upload(files: list[UploadFile] = File(...)):
    saved = []
    for f in files:
        suffix = Path(f.filename).suffix.lower()
        if suffix not in IMG_EXTS:
            continue
        outp = WORKSPACE / "uploads" / f.filename
        outp = unique_path(outp)
        async with aiofiles.open(outp, 'wb') as out_file:
            content = await f.read()
            await out_file.write(content)
        saved.append(outp.name)
    return {"saved": saved}

@app.post("/api/queue_from_uploads")
async def queue_from_uploads(folder_label: str = Form(...), suffix_text: str = Form(""), no_limit: bool = Form(False)):
    """
    Moves files in workspace/uploads to workspace/tudong_input and creates renaming & report rows similar to desktop app.
    """
    state = load_state()
    day = today_dd_mm()
    uploads_dir = WORKSPACE / "uploads"
    tudong_input = WORKSPACE / "tudong_input"
    output_dir = WORKSPACE / "output_images"
    error_dir = WORKSPACE / "error_images"
    input_dir = WORKSPACE / "input_images"
    report_rows = []
    all_files = [p for p in uploads_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not all_files:
        return {"queued": 0}
    for img_path in sorted(all_files):
        orig_name = img_path.name
        if orig_name in state.get("processed_files", []):
            continue
        qr_count, layout = qr_detector.count_qr_codes(img_path)
        state["global_index"] = state.get("global_index", 0) + 1
        gidx = state["global_index"]
        state["folder_index"].setdefault(folder_label, 0)
        state["folder_index"][folder_label] += 1
        idx_local = state["folder_index"][folder_label]
        new_base = f"{gidx}.{day} {folder_label} {idx_local}_{qr_count}"
        if suffix_text:
            new_base += f"_{suffix_text}"
        new_name = new_base + img_path.suffix.lower()
        new_path = unique_path(img_path.with_name(new_name))
        # rename in uploads folder
        try:
            img_path.rename(new_path)
            dest = unique_path(tudong_input / new_path.name)
            shutil.move(str(new_path), str(dest))
            state.setdefault("processed_files", []).append(orig_name)
            report_rows.append([day, gidx, "", "", dest.name])
        except Exception as e:
            # silently continue
            print("Rename/move error:", e)
    save_state(state)
    # copy to input_images for pipeline
    for f in input_dir.glob("*"):
        try:
            f.unlink()
        except Exception:
            pass
    for f in tudong_input.glob("*"):
        if f.is_file():
            shutil.copy2(str(f), str(input_dir / f.name))
    # write report
    report_file = REPORT_DIR / f"report_{day}.csv"
    import csv
    newf = not report_file.exists()
    with open(report_file, "a", newline="", encoding="utf-8") as rf:
        w = csv.writer(rf)
        if newf:
            w.writerow(["Ngày", "STT", "Customer Code", "Task ID", "Tên File"])
        w.writerows(report_rows)
    return {"queued": len(report_rows)}

# start pipeline endpoint (runs processing in background thread)
@app.post("/api/start_pipeline")
async def api_start_pipeline(no_limit: bool = Form(False), rotate_back: bool = Form(True), check_rotate: bool = Form(True)):
    def background_job():
        try:
            input_dir = WORKSPACE / "input_images"
            output_dir = WORKSPACE / "output_images"
            error_dir = WORKSPACE / "error_images"
            completed_dir = WORKSPACE / "completed"
            workers = min(6, max(2, (os.cpu_count() or 4) - 1))
            total = len([p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()])
            sent = {"type": "start", "total": total}
            import asyncio
            # notify start
            loop = asyncio.new_event_loop()
            loop.run_until_complete(manager.send_json(sent))
            def progress_cb(c, t):
                import asyncio
                loop = asyncio.new_event_loop()
                loop.run_until_complete(manager.send_json({"type":"progress","current":c,"total":t}))
            def log_cb(msg):
                import asyncio
                loop = asyncio.new_event_loop()
                loop.run_until_complete(manager.send_json({"type":"log","message":msg}))
            run_tudong_on_folder_sync(input_dir, output_dir, error_dir, workers, no_limit, rotate_back, check_rotate, progress_cb=progress_cb, log_cb=log_cb)
            # move outputs to completed
            moved = 0
            for f in output_dir.glob("*"):
                if f.is_file():
                    try:
                        shutil.move(str(f), str(completed_dir / f.name))
                        moved += 1
                    except Exception as e:
                        loop.run_until_complete(manager.send_json({"type":"log","message":f"Move lỗi {f.name}: {e}"}))
            loop.run_until_complete(manager.send_json({"type":"done","moved":moved}))
        except Exception as e:
            tb = traceback.format_exc()
            import asyncio
            loop = asyncio.new_event_loop()
            loop.run_until_complete(manager.send_json({"type":"error","message":str(e),"trace":tb}))

    thread = threading.Thread(target=background_job, daemon=True)
    thread.start()
    return {"started": True}

@app.websocket("/ws/progress")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()  # keep alive; frontend can send pings
    except WebSocketDisconnect:
        manager.disconnect(ws)

@app.get("/api/list_completed")
async def list_completed():
    completed_dir = WORKSPACE / "completed"
    files = [p.name for p in completed_dir.iterdir() if p.is_file()]
    return {"files": files}

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    fp = WORKSPACE / "completed" / filename
    if not fp.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(fp), filename=filename)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
