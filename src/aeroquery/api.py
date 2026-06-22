from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from aeroquery.detect import get_detector
from aeroquery.storage import save_detection
from aeroquery.agent import ask_agent
from aeroquery.rag import create_report_from_detections, add_report

app = FastAPI(title="AeroQuery")


class VehicleEntry(BaseModel):
    plate: str
    slot_id: int
    vehicle_type: str
    entry_time: str



@app.get("/")
async def read_root():
    return {"message" : "selamm" }



@app.get("/slots/{slot_id}")
async def read_slot(slot_id: int):
    return {"slot_id": slot_id}


@app.post("/vehicles")
async def create_vehicle(entry: VehicleEntry):
    return {"message": "Arac Kaydedildi", "data": entry}



@app.post("/detect")
async def detect(file: UploadFile):
    contents = await file.read()

    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(contents)
    detector = get_detector()
    detections = detector.predict(temp_path)

    for d in detections:
        save_detection(d.class_name, d.confidence, d.bbox)

    report = create_report_from_detections(detections)
    add_report(report)

    return [
    {"class_name": d.class_name, "confidence": d.confidence, "bbox": d.bbox}
    for d in detections
    ]    

class AgentQuery(BaseModel):
    question: str

@app.post("/agent")
async def agent_endpoint(query: AgentQuery):
    answer = ask_agent(query.question)
    return {"answer" : answer}
