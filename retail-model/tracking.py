import mlflow
import mlflow.sklearn


class TrackML:
    def __init__(self, mlflow_tracking_url, experiment_name, build_number):
        self.mlflow_tracking_url = mlflow_tracking_url
        self.experiment_name = experiment_name
        self.build_number = build_number
        self.use_mlflow_remote_server = mlflow_tracking_url is not None

    def __enter__(self):
        if self.use_mlflow_remote_server:
            mlflow.set_tracking_uri(uri=self.mlflow_tracking_url)
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.build_number)

        print("MLFLOW_TRACKING_URL: ", self.mlflow_tracking_url)
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
