import joblib
import pandas as pd
from fastapi import FastAPI,Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title = 'Churn API')


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get('/data')
def data():
    df = joblib.load(r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Churn_Project\models\data_frame.pkl")
    return Response(df.to_json(orient='records'),media_type='application/json')


model = joblib.load(r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Churn_Project\data\churn_model.pkl")
model_columns = joblib.load(r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Churn_Project\models\model_columns.pkl")

class ChurnBlueprint(BaseModel):
    Age:int
    Days_Since_Last_Login:int
    Customer_Service_Calls:int
    Monthly_Spend:float|int
    Gender_female:int
    Gender_male:int
    Gender_non_binary:int
    Gender_unknown:int
    Region_east:int
    Region_north:int
    Region_south:int
    Region_west:int
    Subscription_Plan_basic:int
    Subscription_Plan_plus:int
    Subscription_Plan_pro:int
    Payment_Method_credit_card:int
    Payment_Method_crypto:int
    Payment_Method_e_wallet:int
    Payment_Method_paypal:int

@app.post('/predict')
def churn_prediction(data: ChurnBlueprint):
    df = pd.DataFrame([data.model_dump()])
    df = df.reindex(columns = model_columns , fill_value = 0)

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return {'Prediction': prediction, 'Probability': probability}
