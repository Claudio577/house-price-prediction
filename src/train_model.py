import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import numpy as np
import joblib
# -----------------------------------------
# 1. Carregar dados
# -----------------------------------------
df = pd.read_csv("../data/train.csv")  # Caminho relativo

# Remover colunas que não ajudam
df = df.drop("Id", axis=1)

# Variável alvo
y = df["SalePrice"]
X = df.drop("SalePrice", axis=1)

# Separação entre variáveis numéricas e categóricas
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

# -----------------------------------------
# 2. Pré-processamento
# -----------------------------------------
preprocessamento = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
], remainder="passthrough")

# -----------------------------------------
# 3. Modelo
# -----------------------------------------
modelo = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

pipeline = Pipeline([
    ("prep", preprocessamento),
    ("model", modelo)
])

# -----------------------------------------
# 4. Treino e teste
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

# -----------------------------------------
# 5. Avaliação
# -----------------------------------------
pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("RMSE:", rmse)

# -----------------------------------------
# 6. Exemplo de previsão com um imóvel
# -----------------------------------------
exemplo = X_test.iloc[0:1]
preco_previsto = pipeline.predict(exemplo)[0]

# Salvar o modelo treinado

joblib.dump(pipeline, "../model.pkl")

print("Modelo salvo como model.pkl")
