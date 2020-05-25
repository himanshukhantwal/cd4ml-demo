### Setup project
```
pip3 install virtualenv
virtualenv --python=python3 .venv
pip3 install -r requirements.txt
```



###Model tracking
```
mlflow.log_param("max_depth", max_depth)

mlflow.log_metric("RMSE", rmse)

mlflow.sklearn.log_model(pipeline_model, "model")

mlflow.log_artifact("<some plot file>.png")
```