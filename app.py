import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


# ====================================================
# 📌 CARREGAR DATASET
# ====================================================
df = pd.read_csv("data/train.csv")


# ====================================================
# 📌 TREINAR O MODELO DENTRO DO STREAMLIT
# (Evita erro de versão de pickle)
# ====================================================

st.sidebar.info("🔄 Treinando o modelo... (apenas na primeira carga)")

# Features
X = df.drop(["SalePrice", "Id"], axis=1)
y = df["SalePrice"]

cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
], remainder="passthrough")

modelo = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

pipeline = Pipeline([
    ("prep", preprocess),
    ("model", modelo)
])

pipeline.fit(X, y)


# ====================================================
# 📌 MENU LATERAL
# ====================================================
st.sidebar.title("🏠 Navegação")
pagina = st.sidebar.radio(
    "Ir para:",
    [
        "🔮 Predição de Preço",
        "📊 EDA (Análise Exploratória)",
        "📈 Importância das Features",
        "📘 Sobre o Dataset",
    ]
)


# ====================================================
# 🔮 PÁGINA 1 — PREDIÇÃO
# ====================================================
if pagina == "🔮 Predição de Preço":
    st.title("🏡 Previsão de Preço de Imóvel")

    neighborhoods = sorted(df["Neighborhood"].unique())
    kitchen_qual_list = sorted(df["KitchenQual"].unique())

    # Entradas do usuário
    overall_qual = st.slider("Qualidade geral (1 a 10)", 1, 10, 5)
    gr_liv_area = st.number_input("Área útil (GrLivArea)", 300, 6000, 1500)
    garage_cars = st.slider("Garagem (carros)", 0, 5, 1)
    total_bsmt_sf = st.number_input("Área do porão (TotalBsmtSF)", 0, 3000, 800)
    year_built = st.number_input("Ano de construção", 1870, 2020, 1998)
    year_remod = st.number_input("Ano da reforma", 1950, 2020, 2005)
    bairro = st.selectbox("Bairro (Neighborhood)", neighborhoods)
    kitchen_qual = st.selectbox("Qualidade da cozinha (KitchenQual)", kitchen_qual_list)

    if st.button("Prever preço"):
    
        # Pegamos uma linha REAL do dataset
        entrada = X.iloc[[0]].copy()

        # Substituímos SOMENTE os valores que o usuário informou
        entrada["OverallQual"] = overall_qual
        entrada["GrLivArea"] = gr_liv_area
        entrada["GarageCars"] = garage_cars
        entrada["TotalBsmtSF"] = total_bsmt_sf
        entrada["YearBuilt"] = year_built
        entrada["YearRemodAdd"] = year_remod
        entrada["Neighborhood"] = bairro
        entrada["KitchenQual"] = kitchen_qual

        # Previsão segura, sem erro de categorias
        preco = pipeline.predict(entrada)[0]

        st.success(f"💰 Preço estimado: **${preco:,.2f}**")


# ====================================================
# 📊 PÁGINA 2 — EDA
# ====================================================
elif pagina == "📊 EDA (Análise Exploratória)":
    st.title("📊 EDA — Análise Exploratória")

    st.subheader("Distribuição dos Preços")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["SalePrice"], bins=40, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlação com o Preço")
    corr = df.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False)
    st.bar_chart(corr.head(15))

    st.subheader("GrLivArea x Preço")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.scatter(df["GrLivArea"], df["SalePrice"], alpha=0.5)
    st.pyplot(fig2)


# ====================================================
# 📈 PÁGINA 3 — IMPORTÂNCIA DAS FEATURES
# ====================================================
elif pagina == "📈 Importância das Features":
    st.title("📈 Importância das Variáveis")

    try:
        importances = pipeline.named_steps["model"].feature_importances_
        nomes = pipeline.named_steps["prep"].get_feature_names_out()

        df_imp = pd.DataFrame({"feature": nomes, "importance": importances})
        df_imp = df_imp.sort_values(by="importance", ascending=False).head(20)

        st.bar_chart(df_imp.set_index("feature"))
    except:
        st.error("O modelo não possui informações de importância de features.")


# ====================================================
# 📘 PÁGINA 4 — SOBRE O DATASET
# ====================================================
elif pagina == "📘 Sobre o Dataset":
    st.title("📘 Sobre o Dataset")

    st.write("""
    Este projeto utiliza o dataset **House Prices – Advanced Regression Techniques** do Kaggle.

    O objetivo é prever o preço final de venda de casas na cidade de Ames, Iowa (EUA).
    """)
    
    st.subheader("Primeiras linhas")
    st.dataframe(df.head())

    st.subheader("Estatísticas")
    st.write(df.describe())
