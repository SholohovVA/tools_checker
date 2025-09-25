from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import uuid
from .utils import detect_objects, save_image

app = FastAPI(title="Детекция инструментов (Ultralytics)")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")

sessions = {}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "taken": {"detections": [], "image_url": None, "original_url": None},
        "returned": {"detections": [], "image_url": None, "original_url": None}
    }
    return templates.TemplateResponse("index.html", {
        "request": request,
        "session_id": session_id
    })

@app.post("/upload_original/{kind}/{session_id}")
async def upload_original(kind: str, session_id: str, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    filename = f"orig_{kind}_{session_id}_{file.filename}"
    img_url = save_image(image, filename)

    sessions[session_id][kind]["original_url"] = img_url
    return {"image_url": img_url}

@app.post("/detect/{kind}/{session_id}")
async def detect(kind: str, session_id: str, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    detections, rendered_img = detect_objects(image)

    filename = f"det_{kind}_{session_id}_{file.filename}"
    img_url = save_image(rendered_img, filename)

    sessions[session_id][kind] = {
        "detections": detections,
        "image_url": img_url,
        "original_url": sessions[session_id][kind].get("original_url")
    }

    return {"detections": detections, "image_url": img_url}

@app.get("/compare/{session_id}")
async def compare(session_id: str):
    if session_id not in sessions:
        return {"error": "Session not found"}

    taken = sessions[session_id]["taken"]["detections"]
    returned = sessions[session_id]["returned"]["detections"]

    taken_labels = [d["label"] for d in taken]
    returned_labels = [d["label"] for d in returned]

    all_tools = set(taken_labels + returned_labels)
    summary = []
    for tool in all_tools:
        taken_count = taken_labels.count(tool)
        returned_count = returned_labels.count(tool)
        # Итог = сдано − взято
        summary.append({
            "tool": tool,
            "taken": taken_count,
            "returned": returned_count,
        })

    return {
        "taken_image": sessions[session_id]["taken"]["image_url"],
        "returned_image": sessions[session_id]["returned"]["image_url"],
        "summary": summary
    }

@app.post("/api/detect")
async def api_detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    detections, _ = detect_objects(image)
    # Возвращаем отображаемые названия
    return {"detections": [{"label": d["label"], "confidence": d["confidence"], "bbox": d["bbox"]} for d in detections]}