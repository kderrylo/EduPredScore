import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import sklearn
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt

"""
# ML Project - Student Performance score prediction

Anggota:
- Bryan Mulia
- Jasson Widiarta
- Kasimirus Derryl Odja
- Joel Wilson Suwanto
- Irving Masahiro Samin

Repository: [Github](https://github.com/Jasson9/ml-project)
"""
st.sidebar.header("Processing Options")
model_option = st.sidebar.selectbox("Model:", {"Random Forest":"RandomForest.pkl", "SVM":"SVM.pkl", "Gradient Boosting":"gradient_boosting.pkl"})
scaler_option = st.sidebar.selectbox("Scaler Option:", {"Standard Scaler":"standard.pkl", "Robust Scaler":"robust.pkl", "MinMax Scaler":"minmax.pkl"})


def random_button_callback(set_value_only=False):
    state = st.session_state;
    input_df =pd.read_csv('student-mat.csv', sep=';', usecols=['sex', 'age', 'address', 'Medu', 'Fedu', 
     'traveltime', 'failures', 'paid', 'higher', 'internet','goout', 'G1', 'G2','G3'])
    sample = input_df.sample()
    state = st.session_state;
    state.age = sample['age'].values[0]
    state.sex_idx = list(sex_options.values()).index(sample['sex'].values[0])
    state.address_idx = list(address_options.values()).index(sample['address'].values[0])
    state.Medu_idx = list(Medu_options.values()).index(sample['Medu'].values[0])
    state.Fedu_idx = list(Fedu_options.values()).index(sample['Fedu'].values[0])
    state.traveltime_idx = list(traveltime_options.values()).index(sample['traveltime'].values[0])
    state.failures_idx = list(failures_options.values()).index(sample['failures'].values[0])
    state.paid_idx = list(paid_options.values()).index(sample['paid'].values[0])
    state.higher_idx = list(higher_options.values()).index(sample['higher'].values[0])
    state.internet_idx = list(internet_options.values()).index(sample['internet'].values[0])
    state.goout = sample['goout'].values[0]
    state.G1 = sample['G1'].values[0]
    state.G2 = sample['G2'].values[0]

    if not set_value_only:
        predict(sample.drop(['G3'],axis=1),model_options[model_option], scaler_options[scaler_option])
        state.actual_result = sample['G3'].values[0]

scaler_options ={"Standard Scaler":"standard.pkl", "Robust Scaler":"robust.pkl", "MinMax Scaler":"minmax.pkl"}
model_options = {"Random Forest":"RandomForest.pkl", "SVM":"SVM.pkl", "Gradient Boosting":"gradient_boosting.pkl"}
sex_options = {"Female":"F", "Male": "M"}
address_options = {"Urban":"U", "Rural": "R"}
Medu_options = {"None":0, "Primary Education 4th grade": 1, "5th to 9th grade": 2, "Secondary Education": 3, "Higher Education": 4}
Fedu_options = {"None":0, "Primary Education 4th grade": 1, "5th to 9th grade": 2, "Secondary Education": 3, "Higher Education": 4}
traveltime_options = {"<15 min":1, "15-30 min":2, "30-60 min":3, ">60 min":4}
failures_options = {0:0, 1:1, 2:2, 3:3}
paid_options = {"Yes":"yes", "No":"no"}
higher_options = {"Yes":"yes", "No":"no"}
internet_options = {"Yes":"yes", "No":"no"}

def main_render():
    state = st.session_state;
    st.sidebar.header("Parameters")
    sex = st.sidebar.selectbox("Sex:", sex_options.keys(), 0 if 'sex_idx' not in state else state.sex_idx)
    age = st.sidebar.slider("Age", 6,  24, 6 if 'age' not in state else state.age)
    address = st.sidebar.selectbox("Address:", address_options.keys(), 0 if 'address_idx' not in state else state.address_idx)
    Medu = st.sidebar.selectbox("Mother's education level:", Medu_options.keys(), 0 if 'Medu_idx' not in state else state.Medu_idx)
    Fedu = st.sidebar.selectbox("Father's education level:", Fedu_options.keys(), 0 if 'Fedu_idx' not in state else state.Fedu_idx)
    traveltime = st.sidebar.selectbox("Travel time to school:", traveltime_options.keys(), 0 if 'traveltime_idx' not in state else state.traveltime_idx)
    failures = st.sidebar.selectbox("Number of past class failures:", failures_options.keys(), 0 if 'failures_idx' not in state else state.failures_idx)
    paid = st.sidebar.selectbox("Extra paid classes within the course subject:", paid_options.keys(), 0 if 'paid_idx' not in state else state.paid_idx)
    higher = st.sidebar.selectbox("Wants to take higher education:", higher_options.keys(), 0 if 'higher_idx' not in state else state.higher_idx)
    internet = st.sidebar.selectbox("Internet access at home:", internet_options.keys(), 0 if 'internet_idx' not in state else state.internet_idx)
    goout = st.sidebar.slider("Going out with friends", 1, 5, 1 if 'goout' not in state else state.goout)
    G1 = st.sidebar.number_input("G1",0,30, 0 if 'G1' not in state else state.G1)
    G2 = st.sidebar.number_input("G2",0,30, 0 if 'G2' not in state else state.G2)
    input_df = pd.DataFrame({
        'sex' : [sex_options[sex]],
        'age' : [age],
        'address' : [address_options[address]],
        'Medu' : [Medu_options[Medu]],
        'Fedu' : [Fedu_options[Fedu]],
        'traveltime' : [traveltime_options[traveltime]],
        'failures' : [failures_options[failures]],
        'paid' : [paid_options[paid]],
        'higher' : [higher_options[higher]],
        'internet' : [internet_options[internet]],
        'goout' : [goout],
        'G1' : [G1],
        'G2' : [G2]
    })
    st.sidebar.button('predict', on_click=lambda: predict(input_df,model_options[model_option], scaler_options[scaler_option]))
    st.sidebar.button('predict on all models', on_click=lambda: pred_all_models(input_df, scaler_options[scaler_option]))
    st.sidebar.button('random predict', on_click=random_button_callback)
    if 'result' in st.session_state:
        st.write("prediction result (G3): ",st.session_state.result)
    # if 'actual_result' in st.session_state and st.session_state.actual_result is not None:
    #     st.write("actual result: ",st.session_state.actual_result)
        

def pred_all_models(input_df, scaler_option):
    st.session_state.y_pred = []
    result = {}
    for model_option in model_options:
        pred = predict(input_df, model_options[model_option], scaler_option)
        st.session_state.y_pred.append(pred)
        result[model_option] = pred
    st.session_state.result = "\n".join([f"{k}: {v}" for k, v in result.items()])
    show_plot()

def show_plot():
    # Membuat plot hasil prediksi
    y_pred = st.session_state.y_pred
    fig1, ax1 = plt.subplots()
    ax1.bar(model_options.keys(), y_pred)
    ax1.set_ylabel('G3')
    st.pyplot(fig1)

def predict(param_df,model_option,scaler_option):
    input_df = param_df.copy()
    print(model_option, scaler_option)
    if 'model_option' not in st.session_state or st.session_state.model_option != model_option:
        try:
            # st.session_state.model = joblib.load("Models/"+model_option)
            st.session_state.model = pickle.load(open("Models/"+model_option, 'rb'))
        except:
            st.session_state.model = joblib.load("Models/"+model_option)
        st.session_state.model_option = model_option
    if 'scaler_option' not in st.session_state or st.session_state.scaler_option != scaler_option:
        st.session_state.scaler = pickle.load(open("Scaler/"+scaler_option, 'rb'))
        st.session_state.scaler_option = scaler_option
    print(input_df)
    encoder = LabelEncoder()
    original_df = pd.read_csv('student-mat.csv', sep=';', usecols=['sex', 'age', 'address', 'Medu', 'Fedu', 
     'traveltime', 'failures', 'paid', 'higher', 'internet','goout', 'G1', 'G2'])
    input_df.address=encoder.fit(original_df.address).transform(input_df.address)
    input_df.sex=encoder.fit(original_df.sex).transform(input_df.sex)
    input_df.paid=encoder.fit(original_df.paid).transform(input_df.paid)
    input_df.higher=encoder.fit(original_df.higher).transform(input_df.higher)
    input_df.internet=encoder.fit(original_df.internet).transform(input_df.internet)
    result = input_df[['age', 'Medu', 'Fedu', 
     'traveltime', 'failures', 'G1', 'G2']]
    scaled_df = st.session_state.scaler.transform(result)
    input_df = input_df.copy()
    input_df[result.columns] = scaled_df
    print(input_df)
    print("model:",st.session_state.model)
    pred = st.session_state.model.predict(input_df)
    st.session_state.result = str(pred[0])
    return pred[0]
        
def main():
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        st.session_state.initialized = True
        random_button_callback(True)
    main_render()
    return 0

if __name__ == '__main__':
    main()
