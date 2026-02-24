import os
import shutil
import asyncio
import json
from typing import Dict, List
from fastapi import FastAPI, UploadFile, File, WebSocket, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Import our processing logic
from vrg_bboxmaskpose import VRGConfig, VRGBBoxMaskPose

app = FastAPI(title="BBox Mask Pose UI")

# Object to hold active runners for streaming
active_runners: Dict[str, VRGBBoxMaskPose] = {}

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
UPLOAD_DIR = "data/uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# State for progress updates
processing_status: Dict[str, dict] = {}

class ProgressNotifier:
    def __init__(self, video_id: str, websocket: WebSocket = None):
        self.video_id = video_id
        self.websocket = websocket

    async def notify(self, frame_id: int, total_frames: int, status: str):
        msg = {
            "video_id": self.video_id,
            "frame": frame_id,
            "total": total_frames,
            "status": status,
            "percentage": round((frame_id / total_frames) * 100, 2) if total_frames > 0 else 0
        }
        processing_status[self.video_id] = msg
        if self.websocket:
            await self.websocket.send_json(msg)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"filename": file.filename, "path": file_path}

async def run_processing(video_name: str, video_id: str):
    input_path = os.path.join(UPLOAD_DIR, video_name)
    output_prefix = os.path.join(OUTPUT_DIR, video_name.rsplit('.', 1)[0] + "_vrg")
    
    config = VRGConfig(
        output_dir=OUTPUT_DIR,
        pose_config="checkpoints/rtmpose-23k.py"
    )
    
    runner = VRGBBoxMaskPose(config)
    active_runners[video_id] = runner
    
    def on_progress(frame: int, total: int):
        processing_status[video_id].update({
            "frame": frame,
            "total": total,
            "percentage": round((frame / total) * 100, 1) if total > 0 else 0,
            "status": "Processing"
        })

    try:
        await asyncio.to_thread(runner.process_video, input_path, progress_callback=on_progress)
        processing_status[video_id]["status"] = "Complete"
    except Exception as e:
        logger.exception(f"Processing failed: {e}")
        processing_status[video_id]["status"] = f"Error: {str(e)}"
    finally:
        # Keep runner for a few seconds so results can be seen then cleanup
        await asyncio.sleep(5)
        if video_id in active_runners:
            del active_runners[video_id]

@app.get("/stream/{video_id}")
async def stream_processing(video_id: str):
    if video_id not in active_runners:
        raise HTTPException(status_code=404, detail="No active processing for this video")
    
    runner = active_runners[video_id]
    
    async def frame_generator():
        import cv2
        while video_id in active_runners:
            if runner.last_frame is not None:
                _, jpeg = cv2.imencode('.jpg', runner.last_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            await asyncio.sleep(0.05) # ~20 fps capped stream
            
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/process/{video_name}")
async def start_processing(video_name: str, background_tasks: BackgroundTasks):
    video_id = video_name
    processing_status[video_id] = {"status": "Starting", "frame": 0, "total": 0}
    background_tasks.add_task(run_processing, video_name, video_id)
    return {"video_id": video_id, "message": "Processing started"}

@app.get("/status/{video_id}")
async def get_status(video_id: str):
    return processing_status.get(video_id, {"status": "Unknown"})

@app.websocket("/ws/progress/{video_id}")
async def websocket_endpoint(websocket: WebSocket, video_id: str):
    await websocket.accept()
    try:
        while True:
            # Poll status and send to client
            status = processing_status.get(video_id, {"status": "Waiting"})
            await websocket.send_json(status)
            if status["status"] in ["Complete", "Error"]:
                break
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# Serve static files
app.mount("/static/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
