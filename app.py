import pandas as pd
import uvicorn
from fastapi import FastAPI
from Model import CreditModel, CreditSpecies

train_df = pd.read_csv('finalcredit_df.csv')

app = FastAPI()
model = CreditModel()

st.title('Credit dashboard')

@app4.post('/predict')
def predict_species(credit: CreditSpecies):
    data = credit.dict()
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
    uvicorn.run(app, host='127.0.0.1', port=8000)
