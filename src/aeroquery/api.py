from fastapi import FastAPI
from pydantic import BaseModel


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
   