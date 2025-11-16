import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import numpy as np
import joblib

# Carregar dados
df = pd.read_csv("data/train.csv")
df = df.drop("Id", axis=1)

y = df["SalePrice"]
X = df.drop("SalePrice", axis=1)

cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

preprocessamento = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
], remainder="passthrough")

modelo = XGBRegressor(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

pipeline = Pipeline([
    ("prep", preprocessamento),
    ("model", modelo)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print("Novo RMSE:", rmse)

joblib.dump(pipeline, "model.pkl")
print("Modelo XGBoost salvo!")

