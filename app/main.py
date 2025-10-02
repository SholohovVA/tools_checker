import io
import uuid

import uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.identification import verificate_objects, verificate_solo_objects
from app.utils import CLASS_MAPPING, convert_to_serializable
from app.utils import detect_objects, detect_objects_with_meta, save_image, bbox_to_yolo_format

app = FastAPI(title="Детекция инструментов")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")

sessions = {}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "taken": {"detections": [], "image_url": None, "original_url": None},
        "returned": {"detections": [], "image_url": None, "original_url": None}
    }
    # return templates.TemplateResponse("index.html", {
    #     "request": request,
    #     "session_id": session_id
    # })
    return templates.TemplateResponse("index_ugraded.html", {
        "request": request,
        "session_id": session_id
    })


@app.get("/verify-page", response_class=HTMLResponse, include_in_schema=False)
async def verify_page(request: Request):
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "taken": {"detections": [], "image_url": None, "original_url": None},
        "returned": {"detections": [], "image_url": None, "original_url": None}
    }
    return templates.TemplateResponse("verify.html", {
        "request": request,
        "session_id": session_id
    })


@app.post("/upload_original/{kind}/{session_id}", include_in_schema=False)
async def upload_original(kind: str, session_id: str, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    filename = f"orig_{kind}_{session_id}_{file.filename}"
    img_url = save_image(image, filename, is_original=True)

    sessions[session_id][kind]["original_url"] = img_url
    return {"image_url": img_url,
            "original_url": img_url}


@app.post("/detect/{kind}/{session_id}", include_in_schema=False)
async def detect(kind: str, session_id: str, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Сохраняем оригинальное изображение
    original_filename = f"orig_{kind}_{session_id}_{file.filename}"
    print(f"Saving original image: {original_filename}")
    original_url = save_image(image, original_filename, is_original=True)

    # Получаем детекции с полигонами
    detections, rendered_img, segmentation_data = detect_objects(image)
    filename = f"det_{kind}_{session_id}_{file.filename}"
    print(f"Saving processed image: {filename}")
    img_url = save_image(rendered_img, filename, is_original=False)

    sessions[session_id][kind] = {
        "detections": detections,
        "image_url": img_url,
        "original_url": original_url,
        "segmentation_data": segmentation_data
    }
    return {"detections": detections, "image_url": img_url,
            "original_url": original_url}


@app.post("/verify/{session_id}", include_in_schema=False)
async def verify(session_id: str):
    """Верификация объектов между taken и returned"""
    if session_id not in sessions:
        return {"error": "Session not found"}

    taken_data = sessions[session_id]["taken"]
    returned_data = sessions[session_id]["returned"]
    if not taken_data.get("detections") or not returned_data.get("detections"):
        return {"error": "Необходимо сначала обработать оба изображения (taken и returned)"}

    verification_results = verificate_objects(taken_data, returned_data)
    # Сохраняем результаты в сессии
    sessions[session_id]["verification"] = verification_results
    return {"verification_results": verification_results}


@app.post("/verify-page/verify_solo/{session_id}", include_in_schema=False)
async def verify_solo(session_id: str):
    """Верификация объектов между taken и returned"""
    if session_id not in sessions:
        return {"error": "Session not found"}

    taken_data = sessions[session_id]["taken"]
    returned_data = sessions[session_id]["returned"]

    verification_results = verificate_solo_objects(taken_data, returned_data)
    # Сохраняем результаты в сессии
    sessions[session_id]["verification_solo"] = verification_results
    return {"verification_solo": verification_results}


@app.get("/compare/{session_id}", include_in_schema=False)
async def compare(session_id: str):
    if session_id not in sessions:
        return {"error": "Session not found"}

    taken = sessions[session_id]["taken"]["detections"]
    returned = sessions[session_id]["returned"]["detections"]

    taken_labels = [CLASS_MAPPING[d[-1]] for d in taken]
    returned_labels = [CLASS_MAPPING[d[-1]] for d in returned]

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

@app.post("/api/batch-detect", summary="Провести детекцию инструментов", tags=["Детекция"])
async def batch_detect(files: list[UploadFile] = File(...)):
    all_class_ids = set()
    images_result = {}

    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            detections, rendered_image, img_w, img_h = detect_objects_with_meta(image)

            # Сохраняем оригинальное изображение
            original_filename = f"orig_{file.filename}"
            print(f"Saving original image: {original_filename}")
            original_url = save_image(image, original_filename, is_original=True)

            # Сохраняем полученное изображение с полигонами
            filename = f"det_{file.filename}"
            print(f"Saving processed image: {filename}")
            img_url = save_image(rendered_image, filename, is_original=False)

            for det in detections:
                all_class_ids.add(det[-1])

            image_detections = [
                {
                    "class_id": det[-1],
                    "bbox": bbox_to_yolo_format(det[0:4], img_w, img_h)
                }
                for det in detections
            ]
            images_result[file.filename] = image_detections

        except Exception as e:
            images_result[file.filename] = {"error": str(e)}

    classes_dict = {}
    for cid in sorted(all_class_ids):
        display_name = CLASS_MAPPING[cid]
        classes_dict[str(cid)] = display_name

    result = {
        "classes": classes_dict,
        "images": images_result
    }

    # Конвертируем всё в JSON-совместимый формат
    serializable_result = convert_to_serializable(result)
    return JSONResponse(content=serializable_result)

if __name__ == '__main__':
    # uvicorn.run(app, host='10.128.95.2', port=8014)
    uvicorn.run(app, host='0.0.0.0', port=8014)
