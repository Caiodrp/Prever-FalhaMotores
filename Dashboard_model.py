import streamlit as st
import pandas as pd
import shap
import requests
import pickle
import joblib
import tempfile
import os
import numpy as np
import io

from io import BytesIO
from pycaret.classification import load_model
from lightgbm import LGBMClassifier

@st.cache_data
def load_data():
    df = pd.read_csv('https://github.com/Caiodrp/Prever-FalhaMotores/raw/main/Dados/df_treino_dummie.csv')
    return df

@st.cache_resource
def carregar_modelo(url):
    response = requests.get(url)
    modelo = joblib.load(io.BytesIO(response.content))
    return modelo

def main():
    st.title("Modelo Classificação de Falhas de Motores")

    df = load_data()
    url_modelo = 'https://github.com/Caiodrp/Prever-FalhaMotores/raw/main/lgbm.pkl'
    model = carregar_modelo(url_modelo)

    X = df.drop('failure_type', axis=1)

    # Carregar os valores SHAP do arquivo .npz
    response = requests.get('https://github.com/Caiodrp/Prever-FalhaMotores/raw/main/shap_values.npz')
    with np.load(io.BytesIO(response.content), allow_pickle=True) as data:
        shap_values = [data[key] for key in data.keys()]

    class_names = ['Heat Dissipation Failure', 'No Failure', 'Overstrain Failure', 'Power Failure', 'Random Failures', 'Tool Wear Failure']

    st.subheader("Gráfico de Resumo SHAP")
    shap.summary_plot(shap_values, X, class_names=class_names)

    st.subheader("Faça previsões com o seu próprio arquivo .csv")
    uploaded_file = st.file_uploader("Escolha um arquivo .csv", type="csv")
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        predictions = model.predict(input_df)
        st.write(predictions)

if __name__ == "__main__":
    main()
