import streamlit as st
import pandas as pd
import joblib

st.title("🏡 House Price Prediction App")

st.write("Aplicação em desenvolvimento...")

st.write("Carregando dataset:")
df = pd.read_csv("data/train.csv")
st.dataframe(df.head())

