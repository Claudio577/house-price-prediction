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

# Botão
if st.button("Prever preço"):
    model = joblib.load("model.pkl")

    # Criando o DataFrame de entrada
    entrada = pd.DataFrame([{
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "GarageCars": garage_cars,
        "TotalBsmtSF": total_bsmt_sf
    }])

    # Fazer previsão
    preco = model.predict(entrada)[0]

    st.success(f"💰 Preço estimado: **${preco:,.2f}**")
