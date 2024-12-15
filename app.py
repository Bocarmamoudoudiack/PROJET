import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

# Initialisation de l'application Flask
app = Flask(__name__)

# Chargement du modèle sauvegardé avec pickle
with open('model.pkl', 'rb') as model_file:
    model_data = pickle.load(model_file)
    model = model_data["model"]  # Le modèle RandomForestRegressor
    model_features = model_data["features"]  # Liste des colonnes utilisées pour X
   

def predict_price(name, transmission, year, km_driven, engine, max_power, mileage):
    # Initialisation du tableau x avec des zéros
    x = np.zeros(len(model_features), dtype='float32')  # La taille correspond au nombre total de colonnes utilisées par le modèle

    # Assigner les valeurs numériques aux indices fixes
    x[model_features.index('year')] = year
    x[model_features.index('km_driven')] = km_driven
    x[model_features.index('engine')] = engine
    x[model_features.index('max_power')] = max_power
    x[model_features.index('mileage')] = mileage

    # Encodage One-Hot pour 'name'
    if f"name_{name}" in model_features:
        x[model_features.index(f"name_{name}")] = 1

    # Encodage One-Hot pour 'transmission'
    if f"transmission_{transmission}" in model_features:
        x[model_features.index(f"transmission_{transmission}")] = 1

    # Prédiction avec le modèle
    prediction = model.predict([x])[0]
    return float(format(prediction, '.2f'))


# Route principale pour afficher la page HTML
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error_message = None

    if request.method == "POST":
        try:
            # Récupération des données depuis le formulaire HTML
            name = request.form["name"]
            transmission = request.form["transmission"]
            year = int(request.form["year"])
            km_driven = float(request.form["km_driven"])
            engine = float(request.form["engine"])
            max_power = float(request.form["max_power"])
            mileage = float(request.form["mileage"])

            # Appel de la fonction de prédiction
            prediction = predict_price(name, transmission,year, km_driven, engine, max_power,mileage)
        except Exception as e:
            error_message = f"Erreur lors de la prédiction : {str(e)}"

    return render_template("index22.html", prediction=prediction, error_message=error_message)

# Lancement de l'application
if __name__ == "__main__":
    app.run(debug=True, port=8089)  # Remplacez 8080 par le port souhaité


