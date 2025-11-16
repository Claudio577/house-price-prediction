import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("🏡 House Price Prediction (Melhorado)")

st.write("Preencha os dados abaixo para estimar o preço da casa:")

# Carregar dataset para pegar colunas e valores possíveis
df_treino = pd.read_csv("data/train.csv")

# Campos categóricos e valores únicos
neighborhoods = sorted(df_treino["Neighborhood"].unique())
kitchen_qual_list = sorted(df_treino["KitchenQual"].unique())

# ==== CAMPOS DO USUÁRIO ====

overall_qual = st.slider("Qualidade geral (1 a 10)", 1, 10, 5)
gr_liv_area = st.number_input("Área útil (GrLivArea)", min_value=300, max_value=6000, value=1500)
garage_cars = st.slider("Garagem (número de carros)", 0, 5, 1)
total_bsmt_sf = st.number_input("Área do porão (TotalBsmtSF)", 0, 3000, 800)
year_built = st.number_input("Ano de construção (YearBuilt)", 1870, 2020, 1998)
year_remod = st.number_input("Ano da reforma (YearRemodAdd)", 1950, 2020, 2005)
bairro = st.selectbox("Bairro (Neighborhood)", neighborhoods)
kitchen_qual = st.selectbox("Qualidade da Cozinha (KitchenQual)", kitchen_qual_list)

# ==== BOTÃO DE PREVISÃO ====

if st.button("Prever preço"):

    st.write("Carregando modelo...")
    model = joblib.load("model.pkl")

    # Criar estrutura com TODAS as colunas do modelo
    colunas = df_treino.drop(["SalePrice", "Id"], axis=1).columns
    entrada = pd.DataFrame(columns=colunas)
    entrada.loc[0] = 0

    # Preencher as colunas informadas
    entrada["OverallQual"] = overall_qual
    entrada["GrLivArea"] = gr_liv_area
    entrada["GarageCars"] = garage_cars
    entrada["TotalBsmtSF"] = total_bsmt_sf
    entrada["YearBuilt"] = year_built
    entrada["YearRemodAdd"] = year_remod
    entrada["Neighborhood"] = bairro
    entrada["KitchenQual"] = kitchen_qual

    # Prever
    preco = model.predict(entrada)[0]

    st.success(f"💰 Preço estimado: **${preco:,.2f}**")

# ---- GRÁFICOS -----

st.header("📊 Distribuição dos Preços")
fig, ax = plt.subplots()
ax.hist(df_treino["SalePrice"], bins=40)
ax.set_xlabel("Preço")
ax.set_ylabel("Frequência")
st.pyplot(fig)

st.header("📈 Relação entre GrLivArea e Preço")
fig2, ax2 = plt.subplots()
ax2.scatter(df_treino["GrLivArea"], df_treino["SalePrice"], alpha=0.5)
ax2.set_xlabel("GrLivArea")
ax2.set_ylabel("SalePrice")
st.pyplot(fig2)
