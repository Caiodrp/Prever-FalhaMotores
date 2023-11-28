import streamlit as st
import pandas as pd
import requests
import joblib
import plotly.express as px
import io

from io import BytesIO
from pycaret.classification import predict_model

@st.cache_data
def load_data():
    df = pd.read_csv('https://github.com/Caiodrp/Prever-FalhaMotores/raw/main/Dados/df_treino.csv')
    return df

@st.cache_resource
def carregar_modelo(url):
    response = requests.get(url)
    modelo = joblib.load(io.BytesIO(response.content))
    return modelo

def plot_proporcao(df_treino, classe_escolhida):
    cores = {'M': 'blue', 'L': 'red', 'H': 'green'}
    tipos = df_treino['type'].unique()
    data = []
    for tipo in tipos:
        proporcao = len(df_treino[(df_treino['type'] == tipo) & (df_treino['failure_type'] == classe_escolhida)]) / len(df_treino[df_treino['failure_type'] == classe_escolhida])
        data.append({'Tipo': tipo, 'Classe': classe_escolhida, 'Proporção': proporcao})
    df_means = pd.DataFrame(data)
    fig = px.bar(df_means, x='Classe', y='Proporção', color='Tipo', title=f'Proporção de Classes por Tipo da Ferramenta ({classe_escolhida})', labels={'mean': 'Média', 'Classe': 'Classe', 'Tipo': 'Tipo'})
    st.plotly_chart(fig)

def plot_media(df_treino, variavel_escolhida):
    means = df_treino.groupby("failure_type")[variavel_escolhida].mean()
    df_means = pd.DataFrame({variavel_escolhida: means.index, 'mean': means.values})
    fig = px.bar(df_means, x=variavel_escolhida, y='mean', color=variavel_escolhida, title=f'Médias das falhas por {variavel_escolhida}', labels={'mean': 'Média', variavel_escolhida: variavel_escolhida})
    st.plotly_chart(fig)

def main():
    st.set_page_config(page_title='Prever Falhas de Motores', page_icon='⚙️', layout='wide')

    st.markdown("<h1 style='text-align: center;'>Classificação de Falhas em Motores</h1>", unsafe_allow_html=True)

    df_treino = load_data()
    url_modelo = 'https://github.com/Caiodrp/Prever-FalhaMotores/raw/main/lgbm.pkl'
    model = carregar_modelo(url_modelo)

    X = df_treino.drop(['failure_type', 'Unnamed: 0', 'type'], axis=1)  # Removido 'Unnamed: 0' e 'type'

    col1, col2 = st.columns([2,1])

    opcao = col1.radio('Escolha uma opção', ['Categóricas', 'Contínuas'])

    if opcao == 'Categóricas':
        classe_escolhida = col1.selectbox('Escolha uma classe', df_treino['failure_type'].unique())
        with col1:
            plot_proporcao(df_treino, classe_escolhida)
    else:
        variavel_escolhida = col1.selectbox('Escolha uma variável contínua', X.columns)
        with col1:
            plot_media(df_treino, variavel_escolhida)

    col2.subheader("Previsões de falhas em novos motores")
    uploaded_file = col2.file_uploader("Escolha um arquivo .csv", type="csv")
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        predictions = predict_model(model, data=input_df)  # Modificado aqui
        predictions = predictions[['product_id', 'prediction_label']]  # Seleciona apenas as colunas 'product_id' e 'prediction_label'
        col2.write(predictions)
if __name__ == "__main__":
    main()
