from fastapi import FastAPI


app = FastAPI(title="AeroQuery")

@app.get("/")
async def read_root():
    return {"message" : "selamm" }
