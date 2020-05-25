import os

import mlflow
import mlflow.sklearn

MLFLOW_TRACKING_URL = os.getenv('MLFLOW_TRACKING_URL')
EXPERIMENT_NAME = os.getenv('TENANT', 'local')
RUN_LABEL = os.getenv('BUILD_NUMBER', '0')
USE_MLFLOW_REMOTE_SERVER = MLFLOW_TRACKING_URL is not None


class TrackML:
    def __enter__(self):
        if USE_MLFLOW_REMOTE_SERVER:
            mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URL)
        mlflow.set_experiment(EXPERIMENT_NAME)
        mlflow.start_run(run_name=RUN_LABEL)

        print("MLFLOW_TRACKING_URL: ", MLFLOW_TRACKING_URL)
        self.artifact_uri = mlflow.get_artifact_uri()
        print('artifact_uri: ', self.artifact_uri)

        return self

    def __exit__(self, type, value, traceback):
        mlflow.end_run()

    @staticmethod
    def log_param(key, val):
        mlflow.log_param(key, val)

    def log_params(self, ml_params):
        for key, val in ml_params.items():
            self.log_param(key, val)

    @staticmethod
    def log_metrics(metrics):
        for key, val in metrics.items():
            mlflow.log_metric(key, val)

    @staticmethod
    def log_artifact(filename):
        mlflow.log_artifact(filename)

    @staticmethod
    def log_model(model_type, model, model_name):
        if model_type is "sklearn":
            mlflow.sklearn.log_model(model, model_name)
        else:
            print("Can not track model of type:" + model_type)
