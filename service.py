from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from joblib import load

# Load weights and model
with open('w.pkl', 'rb') as f:
    w = load(f)

with open('adaboost_model.pkl', 'rb') as f:
    model = load(f)

# FastAPI app instance
app = FastAPI(title="Load Type Prediction API")


# Pydantic model for request body
class LoadRequest(BaseModel):
    Date_Time: str  # format: 'dd-mm-yyyy HH:MM'
    Usage_kWh: float
    Lagging_Current_Reactive_Power_kVarh: float
    Leading_Current_Reactive_Power_kVarh: float
    CO2: float
    NSM: int


# Preprocessing function
def inference_preprocess(X_val: pd.DataFrame, weekday_means):
    X_val['Date_Time'] = pd.to_datetime(X_val['Date_Time'], format='%d-%m-%Y %H:%M')
    X_val['day'] = X_val['Date_Time'].dt.day
    X_val['month'] = X_val['Date_Time'].dt.month
    X_val['weekday'] = X_val['Date_Time'].dt.dayofweek
    X_val['NSM'] = X_val['Date_Time'].dt.hour * 3600 + X_val['Date_Time'].dt.minute * 60

    X_val = X_val.rename(columns={'Lagging_Current_Reactive.Power_kVarh': 'Lagging_Current_Reactive_Power_kVarh'})
    X_val = X_val.rename(columns={'CO2(tCO2)': 'CO2'})

    columns_to_fill = ['Lagging_Current_Reactive_Power_kVarh',
                       'Leading_Current_Reactive_Power_kVarh',
                       'CO2', 'Usage_kWh']
    for column in columns_to_fill:
        X_val[column].fillna(X_val['weekday'].map(weekday_means[f'{column}_mean']), inplace=True)

    X_val['actual_load_from_formula'] = (
        (X_val['Usage_kWh'] ** 2 +
         abs(X_val['Lagging_Current_Reactive_Power_kVarh'] -
             X_val['Leading_Current_Reactive_Power_kVarh']) ** 2) ** 0.5
    )

    return X_val[['Usage_kWh', 'Lagging_Current_Reactive_Power_kVarh',
                  'Leading_Current_Reactive_Power_kVarh', 'CO2',
                  'NSM', 'day', 'month', 'weekday',
                  'actual_load_from_formula']]


def classify_load(data: pd.DataFrame):
    load_type = model.predict(inference_preprocess(data, w))
    return 'Light_Load' if load_type == 0 else ('Medium_Load' if load_type == 1 else 'Heavy_Load')


@app.post("/predict")
def predict_load(request: LoadRequest):
    # Prepare DataFrame from request
    input_data = pd.DataFrame([{
        'Usage_kWh': request.Usage_kWh,
        'Lagging_Current_Reactive.Power_kVarh': request.Lagging_Current_Reactive_Power_kVarh,
        'Leading_Current_Reactive_Power_kVarh': request.Leading_Current_Reactive_Power_kVarh,
        'CO2(tCO2)': request.CO2,
        'Date_Time': request.Date_Time,
        'NSM': request.NSM
    }])

    # Run prediction
    result = classify_load(input_data)
    return {"Predicted Load Type": result}


@app.get("/")
def root():
    return {"message": "Load Type Prediction API is running."}
