from datetime import datetime
import sys
import os

import warnings
import pickle

import mlflow
import urllib.request


import pandas as pd

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from prefect import flow, task
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

PROFILE=os.getenv("AWS_PROFILE","default")
TRACKING_SERVER_HOST = os.getenv('AWS_HOST')


@task(retries=3, retry_delay_seconds=2)
def read_data(data_file):
    """
    Load and read the datafile
    from project github repo

    """
    path = f"https://raw.githubusercontent.com/carrionalfredo/Pistachio-Classifier/main/data/{data_file}.csv"

    try:
        len(urllib.request.urlopen(path).read()) > 0
    except Exception as error:
        print(f"Exception {error}")
        print(f"The URL {path} not exist")
        print("Exiting script...")
    else:
        df = pd.read_csv(path)

    return df, path


@task()
def prepare_training_data(data, random_state=1):
    """
    Prepare training data and creates train, validation and test sets

    """

    data.columns = data.columns.str.lower()
    data["class"].replace(["Kirmizi_Pistachio", "Siit_Pistachio"], [0, 1], inplace=True)

    df_fulltrain, df_test = train_test_split(
        data, test_size=0.2, random_state=random_state
    )
    df_train, df_val = train_test_split(
        df_fulltrain, test_size=0.25, random_state=random_state
    )

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train["class"].values
    y_val = df_val["class"].values
    y_test = df_test["class"].values

    del df_train["class"]
    del df_val["class"]
    del df_test["class"]

    dict_train = df_train.to_dict(orient="records")
    dict_val = df_val.to_dict(orient="records")
    dict_test = df_test.to_dict(orient="records")

    return dict_train, dict_val, dict_test, y_train, y_val, y_test


@task(retries=3, retry_delay_seconds=2)
def set_MLFLOW_TRACK(experiment_name):
    """
    Prepare training data and creates train, validation and test sets

    """
    tracking_uri = f"http://{TRACKING_SERVER_HOST}:5000"
    mlflow.set_tracking_uri(tracking_uri)

    if mlflow.search_experiments(filter_string=f'name="{experiment_name}"') == []:
        mlflow.create_experiment(experiment_name)

    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    mlflow.set_experiment(experiment_name)

    print(
        f"""MLFlow settings:
          experiment: {experiment_name}
          Tracking URI: {tracking_uri}"""
    )

    return tracking_uri, experiment_name, experiment_id


@flow(retries=3, retry_delay_seconds=2)
def training_model(data, path, experiment, date=datetime.now()):
    tracking_uri, mlflow_experiment, experiment_id = set_MLFLOW_TRACK(
        experiment_name=experiment
    )

    dict_train, dict_val, dict_test, y_train, y_val, y_test = prepare_training_data(
        data
    )

    for C in [1, 2, 4, 8]:
        for max_iter in [20, 50, 100, 200, 500, 1000]:
            for solver in ["lbfgs", "liblinear", "sag", "saga"]:
                with mlflow.start_run(
                    run_name=f"{experiment}_{date.strftime('%Y%m%d_%H%M%S')}"
                ):
                    params = {"max_iter": max_iter, "solver": solver, "C": C}
                    mlflow.log_params(params)

                    pipeline = make_pipeline(
                        DictVectorizer(),
                        LogisticRegression(
                            C=C,
                            solver=solver,
                            max_iter=max_iter,
                            random_state=1,
                            verbose=0,
                            n_jobs=-1,
                        ),
                    )

                    pipeline.fit(dict_train, y_train)

                    val_y_pred = pipeline.predict(dict_val)
                    val_auc = roc_auc_score(y_val, val_y_pred)
                    test_y_pred = pipeline.predict(dict_test)
                    test_auc = roc_auc_score(y_test, test_y_pred)

                    print(
                        f"""Validation
                                Parameters:
                                {params}
                                Validation AUC Score:
                                {val_auc}
                                Test AUC Score:
                                {test_auc}"""
                    )

                    mlflow.log_metric("val_AUC", val_auc)
                    mlflow.log_metric("test_AUC", test_auc)

                    mlflow.sklearn.log_model(pipeline, artifact_path="model")

        return tracking_uri, mlflow_experiment, experiment_id


@task(retries=3, retry_delay_seconds=2)
def select_best_model(tracking_uri, experiment_ids="1"):
    client = MlflowClient(tracking_uri=tracking_uri)
    best_run = client.search_runs(
        experiment_ids=experiment_ids,  #'1',
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.val_AUC DESC"],
    )

    best_model_run_id = best_run[0].info.run_id
    best_model_auc = best_run[0].data.metrics["val_AUC"]

    print(
        f"""BEST MODEL FOR EXPERIMENT {client.get_experiment(experiment_ids).name}
          run id: {best_run[0].info.run_id},
          parameters: {best_run[0].data.params},
          Validation AUC: {best_model_auc:.4f}"""
    )

    return best_model_run_id, best_model_auc


@task(retries=3, retry_delay_seconds=2)
def register_best_model(run_id, name, auc, datafile):
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(
        model_uri=model_uri,
        name=name,
        tags={"Validation AUC": auc, "Training Datafile": datafile},
    )
    print("""Model {name} registered""")


@flow(retries=3, retry_delay_seconds=2)
def model_stage(tracking_uri, name, run_id, auc):
    client = MlflowClient(tracking_uri=tracking_uri)
    new_model_version = client.search_model_versions(f'run_id="{run_id}"')[0].version

    actual_model_in_prod = client.get_latest_versions(name, stages=["Production"])

    if actual_model_in_prod == []:
        # promote new model to production
        client.transition_model_version_stage(
            name=name,
            version=new_model_version,
            stage="Production",
            archive_existing_versions=True,
        )
        dump_model(run_id=run_id, experiment=name)

        print(f"""Model {name} version {new_model_version} now in Production stage""")

    else:
        prod_model_auc = float(actual_model_in_prod[0].tags["Validation AUC"])
        new_model_auc = auc

        if prod_model_auc > new_model_auc:
            # move new model to staging
            client.transition_model_version_stage(
                name=name, version=new_model_version, stage="Staging"
            )

            print(
                f"""Model in Production has better score than new model registered
                  Model {name} version {new_model_version} was transitioned to Stagin stage"""
            )

        else:
            # promote new model to production
            client.transition_model_version_stage(
                name=name,
                version=new_model_version,
                stage="Production",
                archive_existing_versions=True,
            )

            dump_model(run_id=run_id, experiment=name)

            print(
                f"""Model {name} version {new_model_version} was transitioned to Production stage"""
            )


@task()
def dump_model(run_id, experiment):
    logged_model = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(logged_model)

    pickle.dump(model, open(f"model/{experiment}_model.bin", "wb"))
    print(f"Model saved to model/{experiment}_model.bin")


@flow(retries=3, retry_delay_seconds=2, log_prints=True)
def main(data_file="pistachio_20230724", experiment="Pistachio_Classifier"):
    training_data, path = read_data(data_file=data_file)

    MLFLOW_TRACKING_URI, mlflow_experiment, experiment_id = training_model(
        data=training_data, path=path, experiment=experiment
    )

    best_model, best_model_auc = select_best_model(
        tracking_uri=MLFLOW_TRACKING_URI, experiment_ids=experiment_id
    )

    register_best_model(
        run_id=best_model,
        name=mlflow_experiment,
        auc=best_model_auc,
        datafile=data_file,
    )

    model_stage(
        tracking_uri=MLFLOW_TRACKING_URI,
        name=mlflow_experiment,
        run_id=best_model,
        auc=best_model_auc,
    )

@flow()
def run():
    datafile = sys.argv[1]

    main(data_file=datafile)

if __name__ == "__main__":
    run()
