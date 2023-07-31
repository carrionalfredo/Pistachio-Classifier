# Pistachio Classifier

## Project for MLops ZoomCamp 2023 organized by DataTalksClub

### Objective

The present project has the objective of build an end-to-end ML project for identify the Pistachio type, from a set of 17 parameters, and detect if a sample is an **Kirmizi Pistachio** or **Siit Pistachio**.

### Dataset

The dataset for this project was obtained from Kaggle, in this [link](https://www.kaggle.com/datasets/amirhossei.nmirzaie/pistachio-types-detection). Consist of 1718 rows of 17 columns, 16 numerical parameters and the respective class (type) of pistachio.

The 16 parameters and the corresponding datatype are:

- area, int64
- perimeter, float64
- major_axis, float64
- minor_axis, float64
- eccentricity, float64
- eqdiasq, float64
- solidity, float64
- convex_area, int64
- extent, float64
- aspect_ratio, float64
- roundness, float64
- compactness, float64
- shapefactor_1, float64
- shapefactor_2, float64
- shapefactor_3, float64
- shapefactor_4, float64

![Image from Kaggle](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F11592511%2F9107ea75bea18b095900b48e230bc4ec%2F2.jpg?generation=1688826287210809&alt=media 'Pistachio parameters')

### Train of the model

The notebook with the analysis and preparation of the machine learning model is in this repo [link](https://github.com/carrionalfredo/Pistachio-Classifier/blob/main/project.ipynb).

The pipeline for train, select the best model and promote the model to production is in the **training_aws.py** [file](https://github.com/carrionalfredo/Pistachio-Classifier/blob/main/training_aws.py).

This pipeline uses mlflow and prefect to track and orchestrate the training runs. Also uses AWS EC2 instances with a S3 bucket as artifact storage, and RDS Postgres database as backend store.

The pipeline consists of the following task and flows:

- **read_data** task. Function that reads a .csv file with the dataset for train and test each run, and returns the pandas Dataframe for the next task. These file are in the [**data**](https://github.com/carrionalfredo/Pistachio-Classifier/tree/main/data) folder in this repository.
- **prepare_training_data** task. This function take the Dataframe from **read_data** task and prepares the train, eval and test sets for the training flow.
- **set_MLFLOW_TRACK** task. Funciont that set the mlflow configuration for remote server tracking and also set the experiment **"Pistachio-Classifier"**.
- **training_model** flow. Function that takes the data sets from **prepare_training_data** task, and tracking_uri, mlflow_experiment, experiment_id from **set_MLFLOW_TRACK** task for training and register training runs for a model of Logistic Regression with the following parameters:
    
    - C in [1, 2, 4, 8].
    - max_iter in [20, 50, 100, 200, 500, 1000].
    - solver in ["lbfgs", "liblinear", "sag", "saga"].

    After each run, logs in the remote server the validation and test AUC score, and also log the model.

- **select_best_model** task. Function that select the training run with the highest validation AUC score from the set of runs in the remote server.
- **register_best_model** task. Register in the remote mlflow server the best model selected in the **select_best_model** task.

    ![mlflow runs](https://github.com/carrionalfredo/Pistachio-Classifier/blob/main/images/mlflow_runs.jpg 'mlflow runs')

- **model_stage** flow. This function compare the newest registered model with the actual model in production stage, and select the registered model with greater validation AUC. If the new model registered has a better AUC score, promotes this model to **production** and archives the old. Otherwise, keep actual model in **production** and promotes the new model to **stagin**.

    ![Registered models](https://github.com/carrionalfredo/Pistachio-Classifier/blob/main/images/mlflow_registered_models.jpg 'Registered models in mlflow')

- **dump_model** task. Function that dump the model with a certain **run_id**. When in **model_stage** flow, this task dump the model promoted to production stage, to model [folder](https://github.com/carrionalfredo/Pistachio-Classifier/tree/main/model) in this repository, and replace the old file. The file in this folder, will be the used to load the logistic regression model for prediction runs.
- **main** flow. Main function for the actual training and register pipeline.

![Example of flows and task orchestration in Prefect](https://github.com/carrionalfredo/Pistachio-Classifier/blob/main/images/prefect_flows_tasks.png 'Prefect task and flows orchestration')

For run the training pipeline, set the `PROFILE` and `TRACKING_SERVER_HOST` environmental variables with a valid AWS credentials for profile and EC2 public DNS. After that, run:

```console
pipenv run python training_aws.py [csv datafile]
```
At the moment, is there in the [data](https://github.com/carrionalfredo/Pistachio-Classifier/tree/main/data) folder, the three datafiles used to train and select the actual model in production stage (version 3).

### Deployment

The model can be deployed locally as a flask web service. For this, first build a Docker image with the [Dockerfile](https://raw.githubusercontent.com/carrionalfredo/Pistachio-Classifier/main/Dockerfile):

```console
docker build --rm -t pistachio-classifier-service .
```

Once builded the image, run it in the ` port 9696:9696`:

```console
docker run -it --rm -p 9696:9696  pistachio-classifier-service:latest
```

After the service is running, is alredy posible make predictions for classify pistachio's type. In this repository is there a [script](https://raw.githubusercontent.com/carrionalfredo/Pistachio-Classifier/main/test_webservice.py) for testing the web service. For this run:

```command
python test_webservice.py
```

In the `test_webservice.py` script, is there data for two different samples pistachios in JSON format:

First sample:

```python
data = {
    "area": 73107,
    "perimeter": 1161.8070,
    "major_axis": 442.4074,
    "minor_axis": 217.7261,
    "eccentricity": 0.8705,
    "eqdiasq": 305.0946,
    "solidity": 0.9424,
    "convex_area": 77579,
    "extent": 0.7710,
    "aspect_ratio": 2.0319,
    "roundness": 0.6806,
    "compactness": 0.6896,
    "shapefactor_1": 0.0061,
    "shapefactor_2": 0.0030,
    "shapefactor_3": 0.4756,
    "shapefactor_4": 0.9664
}
```

Second sample:

```python
data = {
    "area": 96395,
    "perimeter": 1352.6740,
    "major_axis": 515.8730,
    "minor_axis": 246.5945,
    "eccentricity": 0.8784,
    "eqdiasq": 350.3340,
    "solidity": 0.9549,
    "convex_area": 100950,
    "extent": 0.7428,
    "aspect_ratio": 2.0920,
    "roundness": 0.6620,
    "compactness": 0.6791,
    "shapefactor_1": 0.0054,
    "shapefactor_2": 0.0026,
    "shapefactor_3": 0.4612,
    "shapefactor_4": 0.9648
}
```

After sending this data to the `pistachio-classifier-service ` web service, you should get the following results, indicating the Model used and the type of pistachio predicted:

```command
{'Model': 'Pistachio_Classifier', 'Type': 'Kirmizi_Pistachio'}
{'Model': 'Pistachio_Classifier', 'Type': 'Siit_Pistachio'}
```

You can modify the `test_webservice.py` script with another set of parameters.

For orchestration purposes, the training pipeline also is deployed in Prefect (see [prefect_daployment.py](https://raw.githubusercontent.com/carrionalfredo/Pistachio-Classifier/main/prefect_deployment.py) script).

### Best Practices

Following the required task, best practices for unit testing, integration testing, code formating, quality cheks, and pre-commit hooks were applied in the elaboration of this project, which are summarized in the [`Makefile`](https://raw.githubusercontent.com/carrionalfredo/Pistachio-Classifier/main/Makefile):

```command
test:
	pytest tests/

quality_checks:
	isort .
	black .
	pylint --recursive=y .

build: quality_checks test
	docker build --rm -t pistachio-classifier-service .

integration_test: build
	LOCAL_IMAGE_NAME=pistachio-classifier-service bash integraton-test/run.sh


setup:
	pipenv install --dev
	pre-commit install
```