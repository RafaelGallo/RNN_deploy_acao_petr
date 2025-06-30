import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from tensorflow.keras.models import load_model

st.set_page_config(page_title="QuantumFinance - Demo Deep Learning", layout="wide")

# --- Configuração dos caminhos dos modelos e scalers ---
CAMINHO_MODELOS_CLF = r"G:\Meu Drive\AI_data_lab\Cursos_ml_AI\Fiap\Deep learning\notebooks\Projeto_final_2\Rede_Neural_FIAP_MBA_QuantumFinance\models\modelos_salvos"
CAMINHO_MODELOS_SERIE = r"G:\Meu Drive\AI_data_lab\Cursos_ml_AI\Fiap\Deep learning\notebooks\Projeto_final_2\Rede_Neural_FIAP_MBA_QuantumFinance\models\modelos_temporais"

TICKERS = ['VALE3', 'BBAS3', 'PETR4', 'CSNA3']

# --- Funções auxiliares ---
@st.cache_resource
def load_dl_model(model_path):
    return load_model(model_path)

@st.cache_resource
def load_scaler(scaler_path):
    return joblib.load(scaler_path)

def predict_serie_temporal(model, scaler, input_data, n_futuro=7, nome_modelo='LSTM'):
    """Faz previsão multistep para série temporal."""
    entrada_atual = input_data.copy()
    previsoes_futuras = []
    for _ in range(n_futuro):
        if nome_modelo == 'CNN2D':
            entrada_mod = entrada_atual.reshape(1, entrada_atual.shape[0], 1, 1)
        else:
            entrada_mod = entrada_atual.reshape(1, entrada_atual.shape[0], entrada_atual.shape[1])
        pred_scaled = model.predict(entrada_mod, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)[0, 0]
        previsoes_futuras.append(pred)
        # Deslizar janela
        nova_amostra = pred_scaled.reshape(1, 1, 1)
        entrada_atual = np.concatenate([entrada_atual[1:], nova_amostra.squeeze()], axis=0)
    return previsoes_futuras

# --- Interface Streamlit ---
st.title("🔮 QuantumFinance - Demo Deep Learning")
st.markdown("""
Bem-vindo ao painel de demonstração de **Redes Neurais para Previsão de Ações**!  
Selecione abaixo o tipo de modelo, o ativo e faça suas previsões.
""")

aba = st.sidebar.radio("Escolha a tarefa:", ['Previsão Série Temporal', 'Classificação'])

# ------------- PREVISÃO SÉRIE TEMPORAL -------------
if aba == 'Previsão Série Temporal':
    st.header("Previsão Série Temporal de Preço de Fechamento")

    ticker = st.selectbox("Escolha o Ativo:", TICKERS)
    modelo_st = st.selectbox("Escolha o Modelo:", ['LSTM', 'GRU', 'CNN1D'])  # Se quiser incluir CNN2D, precisa ajustar reshape

    # Escolha os caminhos corretos para modelo e scaler
    modelo_path = os.path.join(CAMINHO_MODELOS_SERIE, f"{modelo_st}_{ticker}.keras")
    scaler_path = os.path.join(CAMINHO_MODELOS_SERIE, f"scaler_y_{ticker}.pkl")

    # Carregar modelo e scaler
    try:
        model = load_dl_model(modelo_path)
        scaler = load_scaler(scaler_path)
    except Exception as e:
        st.error(f"Erro ao carregar modelo/scaler: {e}")
        st.stop()

    # Entrada do usuário (últimos 15 dias por exemplo)
    st.subheader("Insira os valores dos últimos N dias (features normalizadas ou reais)")
    n_dias = st.number_input("Quantidade de dias passados (conforme seu modelo)", min_value=10, max_value=30, value=15)
    entrada = st.text_area("Cole aqui os preços dos últimos dias, separados por vírgula", value="50, 52, 53, 54, 53, 54, 55, 56, 55, 56, 57, 58, 59, 60, 61")
    n_futuro = st.number_input("Quantos dias à frente deseja prever?", min_value=1, max_value=30, value=7)

    if st.button("Prever série temporal"):
        try:
            ultimos = np.array([float(x) for x in entrada.split(',')]).reshape(n_dias, 1)
            ultimos_scaled = scaler.transform(ultimos)
            ultimos_scaled = ultimos_scaled.reshape(1, ultimos_scaled.shape[0], ultimos_scaled.shape[1])  # batch, time, feat
            previsoes = predict_serie_temporal(model, scaler, ultimos_scaled[0], n_futuro=n_futuro, nome_modelo=modelo_st)
            st.success("Previsão realizada com sucesso!")
            st.line_chart(previsoes)
            st.write("Valores previstos:", np.round(previsoes, 2))
        except Exception as e:
            st.error(f"Erro ao processar entrada: {e}")

# ------------- CLASSIFICAÇÃO -------------
elif aba == 'Classificação':
    st.header("Classificação de Sinal de Compra ou Venda")
    ticker = st.selectbox("Escolha o Ativo:", TICKERS)
    modelo_clf = st.selectbox("Escolha o Modelo:", ['MLP_Profunda', 'MLP_Simples', 'MLP_Shallow'])

    modelo_path = os.path.join(CAMINHO_MODELOS_CLF, f"{ticker}_{modelo_clf}.keras")
    scaler_path = os.path.join(CAMINHO_MODELOS_CLF, f"scaler_{ticker}.pkl")

    try:
        model = load_dl_model(modelo_path)
        scaler = load_scaler(scaler_path)
    except Exception as e:
        st.error(f"Erro ao carregar modelo/scaler: {e}")
        st.stop()

    st.subheader("Insira os dados das features do dia (na mesma ordem do treino)")
    entrada = st.text_area("Cole os valores das features, separados por vírgula", value="50, 0.2, 1.5, 0, 1, 0.1")
    if st.button("Classificar"):
        try:
            X = np.array([float(x) for x in entrada.split(',')]).reshape(1, -1)
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0,0]
            classe = "Compra (+1)" if pred >= 0.5 else "Venda (-1)"
            st.success(f"Sinal Predito: {classe} (score: {pred:.2f})")
        except Exception as e:
            st.error(f"Erro ao processar entrada: {e}")