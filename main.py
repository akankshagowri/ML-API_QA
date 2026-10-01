from typing import Annotated
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from app.model import get_model, LABELS, VERSION

app = FastAPI(title='ML API QA Demo', version='1.0.0')

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc):
    # Non-finite input must not be echoed into a JSON error response.
    details = [{key: error[key] for key in ('loc', 'msg', 'type')}
               for error in exc.errors()]
    return JSONResponse(status_code=422, content={'detail': details})

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    features: list[Annotated[float, Field(strict=True, ge=0, le=20, allow_inf_nan=False)]] = Field(min_length=4, max_length=4)

class PredictionResponse(BaseModel):
    model_version: str
    label: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]

@app.get('/health')
def health():
    return {'status':'ok', 'model_version':VERSION}

@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest):
    probabilities = get_model().predict_proba([request.features])[0]
    index = int(probabilities.argmax())
    return PredictionResponse(model_version=VERSION, label=LABELS[index],
        confidence=float(probabilities[index]),
        probabilities={name:float(value) for name,value in zip(LABELS, probabilities)})
