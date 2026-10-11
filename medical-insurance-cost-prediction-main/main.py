from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model = joblib.load("insurance_model.pkl")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"prediction": None}
    )


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    age: int = Form(...),
    sex: int = Form(...),
    bmi: float = Form(...),
    children: int = Form(...),
    smoker: int = Form(...),
    region: int = Form(...)
):

    df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    prediction = model.predict(df)[0]

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"prediction": round(float(prediction), 2)}
    )