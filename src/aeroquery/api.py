from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from aeroquery.detect import Detector
from aeroquery.storage import save_detection
from aeroquery.agent import ask_agent


app = FastAPI(title="AeroQuery")
detector = Detector("models/best.pt")

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

    detections = detector.predict(temp_path)

    for d in detections:
        save_detection(d.class_name, d.confidence, d.bbox)

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
