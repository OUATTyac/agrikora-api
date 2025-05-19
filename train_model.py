import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("data/cacao_data.csv")
X = df[["pluie", "temperature", "fertilisant", "surface"]]
y = df["rendement"]

model = RandomForestRegressor()
model.fit(X, y)

joblib.dump(model, "models/rf_cacao_model.pkl")