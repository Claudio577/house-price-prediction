import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Carregar dataset
# -------------------------------
df = pd.read_csv("data/train.csv")

# -------------------------------
# MENU LATERAL
# -------------------------------
st.sidebar.title("🏠 Navegação")
pagina = st.sidebar.radio("Ir para:", ["🔮 Predição de Preço", "📊 EDA (Análise Exploratória)"])



# =====================================================================
# 📌 PÁGINA 1 — PREDIÇÃO
# =====================================================================

if pagina == "🔮 Predição de Preço":
    st.title("🏡 Previsão de Preço de Imóvel")
    st.write("Preencha os dados abaixo para estimar o valor da casa.")

    # Lista de valores únicos
    neighborhoods = sorted(df["Neighborhood"].unique())
    kitchen_qual_list = sorted(df["KitchenQual"].unique())

    # Campos do usuário
    overall_qual = st.slider("Qualidade geral (1 a 10)", 1, 10, 5)
    gr_liv_area = st.number_input("Área útil (GrLivArea)", 300, 6000, 1500)
    garage_cars = st.slider("Garagem (número de carros)", 0, 5, 1)
    total_bsmt_sf = st.number_input("Área do porão (TotalBsmtSF)", 0, 3000, 800)
    year_built = st.number_input("Ano de construção (YearBuilt)", 1870, 2020, 1998)
    year_remod = st.number_input("Ano da reforma (YearRemodAdd)", 1950, 2020, 2005)
    bairro = st.selectbox("Bairro (Neighborhood)", neighborhoods)
    kitchen_qual = st.selectbox("Qualidade da Cozinha (KitchenQual)", kitchen_qual_list)

    if st.button("Prever preço"):
        model = joblib.load("model.pkl")

        # Criar dataframe com TODAS as colunas
        colunas = df.drop(["SalePrice", "Id"], axis=1).columns
        entrada = pd.DataFrame(columns=colunas)
        entrada.loc[0] = 0

        # Atualiza colunas preenchidas
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



# =====================================================================
# 📌 PÁGINA 2 — EDA
# =====================================================================

if pagina == "📊 EDA (Análise Exploratória)":
    st.title("📊 Análise Exploratória de Dados (EDA)")
    st.write("Explore insights sobre o conjunto de dados.")

    # ---- Distribuição do preço ----
    st.subheader("Distribuição dos Preços")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["SalePrice"], bins=40, kde=True, ax=ax)
    st.pyplot(fig)

    # ---- Correlação ----
    st.subheader("Correlação com o preço")
    corr = df.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False)
    st.bar_chart(corr.head(15))

    # ---- Scatter GrLivArea ----
    st.subheader("GrLivArea vs SalePrice")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.scatter(df["GrLivArea"], df["SalePrice"], alpha=0.5)
    ax2.set_xlabel("GrLivArea")
    ax2.set_ylabel("SalePrice")
    st.pyplot(fig2)

    # ---- Scatter OverallQual ----
    st.subheader("OverallQual vs SalePrice")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.scatter(df["OverallQual"], df["SalePrice"], alpha=0.5)
    ax3.set_xlabel("OverallQual")
    ax3.set_ylabel("SalePrice")
    st.pyplot(fig3)

    st.info("Você pode adicionar MUITO mais gráficos se quiser. 😉")
