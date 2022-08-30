from flask import Flask
from flask import request
import pandas as pd

from flask_cors import CORS

from model.naiveBayesClassifier import NaiveBayesClassifier
app = Flask(__name__)
CORS(app)

data_training = pd.read_csv('Dataset.csv')
y = data_training['estado_del_credito'].values # valores objetivo como cadena
X = data_training[['valor_del_prestamo', 'ingresos_mensuales', 'egresos_mensuales', 'garantia_del_prestamo']].values # valores de características
y_train = y[:608]
y_val = y[608:]

X_train = X[:608]
X_val = X[608:]


@app.route('/credit', methods=['POST'])
def credit():
    if request.method == 'POST':
        message: dict = request.json if request.json is not None else {}
        data = message['data']
        predict = predict_nbc(X_train, data, y_train)
        message_response = ("Felicidades su credito esta en estado Aprobado"
                            if predict == 'Aprobado' else "Lo sentimos, desafortunadamente su credito esta en estado Rechazado, un asesor se pondra en contacto para evaluar este caso.")
    return {"response": message_response}

@app.route('/accuracy', methods=['GET'])
def accuracy():
    if request.method == 'GET':
        predict = accuracy_predict_nbc(X_train, X_val, y_train, y_val)
    return predict


def predict_nbc(X_train_nbc, X_val_nbc, y_train_nbc):
    ## Crear la instancia de Naive Bayes Classifier con los datos de entrenamiento

    nbc = NaiveBayesClassifier(X_train_nbc, y_train_nbc)
    return nbc.classify(X_val_nbc)

def accuracy_predict_nbc(X_train_nbc, X_val_nbc, y_train_nbc, y_val_nbc):
    ## Crear la instancia de Naive Bayes Classifier con los datos de entrenamiento

    nbc = NaiveBayesClassifier(X_train_nbc, y_train_nbc)


    total_cases = len(y_val_nbc) # tamaño del conjunto de validación

    # Ejemplos bien clasificados y ejemplos mal clasificados
    good = 0
    bad = 0

    for i in range(total_cases):
        predict = nbc.classify(X_val_nbc[i])
        #print(f"{y_val[i]} - {predict}")
        if y_val[i] == predict:
            good += 1
        else:
            bad += 1

    return {
        "total_cases": total_cases,
        "right": good,
        "wrong": bad,
        "accuracy": (good/total_cases)*100
    }