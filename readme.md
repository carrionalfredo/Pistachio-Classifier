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

### Train fo the model

The notebook with the analysis and preparation of the machine learning model is in this repo [link](https://github.com/carrionalfredo/Pistachio-Classifier/blob/main/project.ipynb).

The pipeline for train, select the best model and promote the model to production is in the **training_aws.py** [file](https://github.com/carrionalfredo/Pistachio-Classifier/blob/main/training_aws.py).

This pipeline uses mlflow and prefect to track and orchestrate the training runs. Also uses AWS EC2 instances with a S3 bucket as artifact storage, and RDS Postgres database backend store.

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
- **model_stage** flow. This function compare the newest registered model with the actual model in production stage, and select the registered model with greater validation AUC. If the new model registered has a better AUC score, promotes this model to **production** and archives the old. Otherwise, keep actual model in **production** and promotes the new model to **stagin**.
- **dump_model** task. Function that dump the model with a certain **run_id**. When in **model_stage** flow, this task dump the model promoted to production stage, to model [folder](https://github.com/carrionalfredo/Pistachio-Classifier/tree/main/model) in this repository, and replace the old file. The file in this folder, will be the used to load the logistic regression model for prediction runs.
- **main** flow. Main function for the actual training and register pipeline.

![Example of flows and task orchestration in Prefect](https://github.com/carrionalfredo/Pistachio-Classifier/blob/main/images/prefect_flows_tasks.png 'Prefect task and flows orchestration')