from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, Dict, Optional
from pydantic.v1 import NoneStr

from ml_pipeline import TitanicModel

app = FastAPI()
model = TitanicModel()

class PredictRequest(BaseModel):
    age: Optional[float] = None
    who: Optional[str] = None
    parch: Optional[int] = None
    sibsp: Optional[int] = None
    deck: Optional[str] = None
    fare: Optional[float] = None
    pclass: Optional[int] = None

@app.get("/")
def read_root():
    if model.models:
        keys = '\n'.join(i for i in model.models)
        status = 'Model is fitted:', keys
    else:
        status = 'Model is not fitted'
    return status

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post('/train_model/{model_type}')
def train_model(model_type):
    model.train(model_type)
    return 'Train is ended...'

@app.post('/predict/{model_type}')
def predict(request: PredictRequest, model_type):
    res = model.predict(request, model_type)
    return({'Probability:' : res})