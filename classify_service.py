import os
import pickle
from prefect import flow, task
from flask import Flask, request, jsonify

#experiment_name = os.getenv('EXPERIMENT', 'Pistachio_Classifier')
experiment_name = os.getenv('EXPERIMENT')

@task()
def load_mlflow_model(experiment):
    with open(f'{experiment}_model.bin', 'rb') as f_in:
        model = pickle.load(f_in)
    return model

@flow()
def predict(features, experiment):
    model = load_mlflow_model(experiment=experiment)
    preds = model.predict(features)
    return preds[0]

app = Flask('pistachio-classifier')

@app.route('/classify', methods= ['POST'])
@flow()
def predict_endpoint():
    data = request.get_json()
    pred= predict(data, experiment=experiment_name)

    if int(pred) == 0:
        type = 'Kirmizi_Pistachio'
    else:
        type = 'Siit_Pistachio'

    result = {
        'Type': type,
        'Model': experiment_name,
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug= True, host ='0.0.0.0', port= 9696)