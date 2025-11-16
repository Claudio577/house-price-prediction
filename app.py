import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ====================================================
# CARREGAR DADOS E MODELO
# ====================================================
df = pd.read_csv("data/train.csv")


# ====================================================
# MENU LATERAL
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
    st.write("Preencha os dados abaixo para prever o valor da casa.")

    neighborhoods = sorted(df["Neighborhood"].unique())
    kitchen_qual_list = sorted(df["KitchenQual"].unique())

    # ------ Inputs ------
    overall_qual = st.slider("Qualidade geral (1 a 10)", 1, 10, 5)
    gr_liv_area = st.number_input("Área útil (GrLivArea)", 300, 6000, 1500)
    garage_cars = st.slider("Garagem (carros)", 0, 5, 1)
    total_bsmt_sf = st.number_input("Área do porão (TotalBsmtSF)", 0, 3000, 800)
    year_built = st.number_input("Ano de construção", 1870, 2020, 1998)
    year_remod = st.number_input("Ano da reforma", 1950, 2020, 2005)
    bairro = st.selectbox("Bairro (Neighborhood)", neighborhoods)
    kitchen_qual = st.selectbox("Qualidade da cozinha (KitchenQual)", kitchen_qual_list)


    if st.button("Prever preço"):
        colunas = df.drop(["SalePrice", "Id"], axis=1).columns
        entrada = pd.DataFrame(columns=colunas)
        entrada.loc[0] = 0

        entrada["OverallQual"] = overall_qual
        entrada["GrLivArea"] = gr_liv_area
        entrada["GarageCars"] = garage_cars
        entrada["TotalBsmtSF"] = total_bsmt_sf
        entrada["YearBuilt"] = year_built
        entrada["YearRemodAdd"] = year_remod
        entrada["Neighborhood"] = bairro
        entrada["KitchenQual"] = kitchen_qual

        preco = model.predict(entrada)[0]
        st.success(f"💰 Preço estimado: **${preco:,.2f}**")



# ====================================================
# 📊 PÁGINA 2 — EDA
# ====================================================
if pagina == "📊 EDA (Análise Exploratória)":
    st.title("📊 Análise Exploratória (EDA)")

    st.subheader("Distribuição dos preços")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["SalePrice"], bins=40, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlação das variáveis com o preço")
    corr = df.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False)
    st.bar_chart(corr.head(15))

    st.subheader("GrLivArea x Price")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.scatter(df["GrLivArea"], df["SalePrice"], alpha=0.5)
    ax2.set_xlabel("GrLivArea")
    ax2.set_ylabel("SalePrice")
    st.pyplot(fig2)



# ====================================================
# 📈 PÁGINA 3 — IMPORTÂNCIA DAS FEATURES
# ====================================================
if pagina == "📈 Importância das Features":
    st.title("📈 Importância das Variáveis")

    st.write("Importância calculada pelo modelo XGBoost ou RandomForest.")

    try:
        # extrair importâncias do modelo
        importances = model.named_steps["model"].feature_importances_
        nomes = model.named_steps["prep"].get_feature_names_out()

        df_imp = pd.DataFrame({"feature": nomes, "importance": importances})
        df_imp = df_imp.sort_values("importance", ascending=False).head(20)

        st.bar_chart(df_imp.set_index("feature"))

    except:
        st.warning("Não foi possível extrair importâncias. Modelo pode não suportar.")



# ====================================================
# 📘 PÁGINA 4 — SOBRE O DATASET
# ====================================================
if pagina == "📘 Sobre o Dataset":
    st.title("📘 Sobre o Dataset")

    st.write("""
    Este projeto utiliza o famoso dataset **House Prices – Advanced Regression Techniques** da competição do Kaggle.
    
    **Descrição:**
    - Contém 79 variáveis que descrevem casas em Ames, Iowa, EUA.
    - O objetivo é prever o preço final de venda (SalePrice).
    
    **Algumas informações importantes:**
    - Variáveis numéricas (ex: GrLivArea, YearBuilt)
    - Variáveis categóricas (ex: Neighborhood, KitchenQual)
    - Coluna alvo: **SalePrice**
    
    **Fonte:** https://www.kaggle.com/c/house-prices-advanced-regression-techniques
    """)

    st.subheader("Primeiras linhas do dataset")
    st.dataframe(df.head())

    st.subheader("Informações estatísticas")
    st.write(df.describe())
