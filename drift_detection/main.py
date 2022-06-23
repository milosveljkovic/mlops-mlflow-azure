from typing import Union
from pydantic import BaseModel
from river import drift

from fastapi import FastAPI
import uvicorn

drift_detector = drift.ADWIN()

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None

class Record(BaseModel):
    a: float
    # attitude_roll	: float
    # attitude_pitch	: float
    # attitude_yaw	: float
    # userAcceleration_x	: float
    # userAcceleration_y	: float
    # userAcceleration_z	: float
    # act	: float
    # id	: float
    # weight	: float
    # height	: float
    # age	: float
    # gender	: float
    # trial	: float

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/stream")
def stream(item: Record):
    drift_detector.update(item.a)
    if drift_detector.change_detected:
        # The drift detector indicates after each sample if there is a drift in the data
        print(f'Change detected at index')
        drift_detector.reset()
        return "drift_detected"
    return "no drift yet"
    # return {"item_name": item.attitude_roll}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)