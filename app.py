import streamlit as st
import pandas as pd
import joblib

st.title("🏡 House Price Prediction")

st.write("Preencha os dados abaixo para estimar o preço da casa:")

# Campos de entrada
overall_qual = st.slider("Qualidade geral (1 a 10)", 1, 10, 5)
gr_liv_area = st.number_input("Área útil em pés quadrados (GrLivArea)", 500, 6000, 1500)
garage_cars = st.slider("Número de carros na garagem", 0, 5, 1)
total_bsmt_sf = st.number_input("Área total do porão (TotalBsmtSF)", 0, 5000, 800)

# Carregar colunas originais do dataset
df_treino = pd.read_csv("data/train.csv")
colunas = df_treino.drop(["SalePrice", "Id"], axis=1).columns

if st.button("Prever preço"):
    model = joblib.load("model.pkl")

    # Criando um dataframe com TODAS as colunas que o modelo espera
    entrada = pd.DataFrame(columns=colunas)
    entrada.loc[0] = 0  # inicializa tudo com 0 ou vazio

    # Atualiza SOMENTE as colunas disponíveis no app
    entrada["OverallQual"] = overall_qual
    entrada["GrLivArea"] = gr_liv_area
    entrada["GarageCars"] = garage_cars
    entrada["TotalBsmtSF"] = total_bsmt_sf

    # Prever
    preco = model.predict(entrada)[0]

    st.success(f"💰 Preço estimado: **${preco:,.2f}**")
