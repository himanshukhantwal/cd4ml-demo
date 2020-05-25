import numpy as np
import xgboost as xgb
from sklearn.metrics import *
from sklearn.pipeline import Pipeline


def train_model(train_df, label_col, max_depth, n_trees, lr):
    label_data = train_df[[label_col]]
    training_data = train_df.drop([label_col], axis=1)

    xgb_regressor = xgb.XGBRegressor(
        max_depth=max_depth,
        n_estimators=n_trees,
        learning_rate=lr,
        random_state=42,
        seed=42,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_lambda=1,
        gamma=1)
    pipeline = Pipeline(steps=[("regressor", xgb_regressor)])
    return pipeline.fit(training_data, label_data)


def evaluate_model(pipeline_model, test_df, label_col):
    label_data = test_df[[label_col]]
    test_data = test_df.drop([label_col], axis=1)

    prediction = pipeline_model.predict(test_data)

    return metrics(prediction, label_data)


def metrics(prediction, actual):
    rmse = np.sqrt(mean_squared_error(actual, prediction))
    mae = mean_absolute_error(actual, prediction)
    r2 = r2_score(actual, prediction)
    return rmse, mae, r2
