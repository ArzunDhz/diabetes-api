
import numpy as np
import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
app = FastAPI()

origins = [
    "*","http://localhost:8081"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionData(BaseModel) :
    Pregnancies:int
    Glucose:float
    BloodPressure:float
    SkinThickness:float
    Insulin:float
    BMI:float
    DiabetesPedigreeFunction:float
    Age:int

@app.get('/')
def home():
    return {"Working"}
    
@app.post('/predict')
def predict_diabeties( data : PredictionData):
    print(data)
    loaded_model = pickle.load(open("m.sav",'rb'))
    input_data = (
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    )
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction =  loaded_model.predict(input_data_reshaped)
    if(prediction[0] == 0):
        return {"You dont have daibeties"}
    else:
        return{"You have diabeties"}


