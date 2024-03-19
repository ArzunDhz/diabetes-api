
import numpy as np
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

class PredictionData(BaseModel) :
    Pregnancies:int
    Glucose:float
    BloodPressure:float
    SkinThickness:float
    Insulin:float
    BMI:float
    DiabetesPedigreeFunction:float
    Age:int
    
    
@app.post('/predict')
def predict_diabeties( data : PredictionData):
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


