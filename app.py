from flask import Flask, request, jsonify, send_file
import joblib
import numpy as np
import os
from fpdf import FPDF

app = Flask(__name__)
model = joblib.load("models/rf_cacao_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return "API AGRIKORA avec IA, alerte et rapport PDF"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    X = np.array([
        data["pluie"],
        data["temperature"],
        data["fertilisant"],
        data["surface"]
    ]).reshape(1, -1)
    prediction = model.predict(X)[0]
    alert = None
    if prediction < 1.5:
        alert = "Alerte : rendement estimé faible. Risque à surveiller."
    return jsonify({
        "rendement_prevu": round(prediction, 2),
        "alerte": alert
    })

@app.route("/rapport", methods=["POST"])
def rapport():
    data = request.get_json()
    pluie = data["pluie"]
    temperature = data["temperature"]
    fertilisant = data["fertilisant"]
    surface = data["surface"]
    prediction = model.predict(np.array([pluie, temperature, fertilisant, surface]).reshape(1, -1))[0]

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Rapport AGRIKORA - Rendement Cacao", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, f"Pluie : {pluie} mm", ln=True)
    pdf.cell(200, 10, f"Température : {temperature} °C", ln=True)
    pdf.cell(200, 10, f"Fertilisant : {fertilisant} kg/ha", ln=True)
    pdf.cell(200, 10, f"Surface : {surface} ha", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, f"Rendement estimé : {round(prediction, 2)} t/ha", ln=True)
    if prediction < 1.5:
        pdf.set_text_color(255, 0, 0)
        pdf.cell(200, 10, "⚠ Alerte : rendement faible", ln=True)
    pdf.output("rapport_agrikora.pdf")
    return send_file("rapport_agrikora.pdf", as_attachment=True)