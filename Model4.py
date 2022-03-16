import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel
import joblib
import streamlit as st
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import shap

url='https://github.com/Chmnx/P7-Credit/blob/main/creditdf.csv'
train_df = pd.read_csv('creditdf.csv')

import uvicorn
from fastapi import FastAPI
from Model4 import IrisModel, IrisSpecies
import streamlit as st
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

app4 = FastAPI()
model = IrisModel()

st.title('Credit dashboard')

@app4.post('/predict')
def predict_species(iris: IrisSpecies):
    data = iris.dict()
    prediction, probability = model.predict_species(data['AGE'],
                                                    data['DAYS_EMPLOYED'],
                                                    data['DAYS_ID_PUBLISH'],
                                                    data['REGION_RATING_CLIENT'],
                                                    data['REG_CITY_NOT_WORK_CITY'],
                                                    data['EXT_SOURCE_2'],
                                                    data['EXT_SOURCE_3'],
                                                    data['DAYS_LAST_PHONE_CHANGE'],
                                                    data['CODE_GENDER_F'],
                                                    data['Working'],
                                                    data['Higher_Education'],
                                                    data['AMT_CREDIT'],
                                                    data['AMT_INCOME_TOTAL'],
                                                    data['AMT_ANNUITY'])
    return {'prediction': prediction,'probability': probability}


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app4, host='127.0.0.1', port=8000)
