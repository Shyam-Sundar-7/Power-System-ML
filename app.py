import streamlit as st
import pandas as pd
from joblib import load

# import pickle

with open('w.pkl', 'rb') as f:
    w = load(f)

with open('adaboost_model.pkl', 'rb') as f:
    model =load(f)


def inference_preprocess(X_val,weekday_means):
    X_val['Date_Time'] = pd.to_datetime(X_val['Date_Time'], format='%d-%m-%Y %H:%M')
    X_val['day'] = X_val['Date_Time'].dt.day
    X_val['month']=X_val['Date_Time'].dt.month
    X_val['weekday'] = X_val['Date_Time'].dt.dayofweek
    X_val['NSM'] =  X_val['Date_Time'].dt.hour * 3600 + X_val['Date_Time'].dt.minute * 60

    X_val = X_val.rename(columns={'Lagging_Current_Reactive.Power_kVarh': 'Lagging_Current_Reactive_Power_kVarh'})
    # Rename column 'CO2(tCO2)' to 'CO2'
    X_val = X_val.rename(columns={'CO2(tCO2)': 'CO2'})
    # Scale column 'NSM_calculated' between 0 and 1


    columns_to_fill=['Lagging_Current_Reactive_Power_kVarh','Leading_Current_Reactive_Power_kVarh','CO2','Usage_kWh']
    for column in columns_to_fill:
        X_val[column].fillna(X_val['weekday'].map(weekday_means[f'{column}_mean']), inplace=True)
    
    X_val['actual_load_from_formula']=(X_val['Usage_kWh']**2+abs(X_val['Lagging_Current_Reactive_Power_kVarh']-X_val['Leading_Current_Reactive_Power_kVarh'])**2)**0.5

    return X_val[['Usage_kWh', 'Lagging_Current_Reactive_Power_kVarh',
       'Leading_Current_Reactive_Power_kVarh', 'CO2', 'NSM', 'day', 'month',
       'weekday', 'actual_load_from_formula']]
    

# Load the pre-trained model
def classify_load(data):
    # Preprocess input data
    # Predict load type
    load_type = model.predict(inference_preprocess(data, w))

    return 'Light_Load' if load_type==0 else ('Medium_Load' if load_type==1 else 'Heavy_Load')

# Streamlit UI
st.title('Load Type Prediction')
st.write('Enter the following data to predict the load type:')

# Input fields
date_time = st.text_input('Date_Time(format: dd-mm-yyyy HH:MM)', '26-02-2018 17:40')
usage_kwh = st.number_input('Usage_kWh', value=0.0, step=0.01)
lagging_reactive_power = st.number_input('Lagging_Current_Reactive.Power_kVarh', value=0.0)
leading_reactive_power = st.number_input('Leading_Current_Reactive_Power_kVarh', value=0.0)
co2 = st.number_input('CO2(tCO2)', value=0.0, step=0.01)
nsm = st.number_input('NSM', value=0)

if st.button('Submit'):
    # Prepare input data
    input_data = pd.DataFrame({
        'Usage_kWh': [usage_kwh],
        'Lagging_Current_Reactive.Power_kVarh': [lagging_reactive_power],
        'Leading_Current_Reactive_Power_kVarh': [leading_reactive_power],
        'CO2(tCO2)': [co2],
        'Date_Time': [date_time],
        'NSM': [nsm]
    })
    # Classify load type
    load_type = classify_load(input_data)
    st.write('Predicted Load Type:', load_type)