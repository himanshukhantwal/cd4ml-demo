import click
import mlflow
import mlflow.sklearn

import data_reader as reader
import model as md


@click.command()
@click.option("--data")
@click.option("--label_col")
@click.option("--max_depth", default=7)
@click.option("--n_trees", default=200)
@click.option("--lr", default=0.005)
def main(data, label_col, max_depth, n_trees, lr):
    test_data, train_data = reader.prepare_data(data)

    pipeline_model = md.train_model(train_data, label_col, max_depth, n_trees, lr)

    mlflow.sklearn.log_model(pipeline_model, "model")

    rmse, mae, r2 = md.evaluate_model(pipeline_model, test_data, label_col)

    print("Model tree model (max_depth=%f, trees=%f, lr=%f):" % (max_depth, n_trees, lr))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_trees", n_trees)
    mlflow.log_param("lr", lr)

    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("MAE", mae)

    print("Model saved in run %s" % mlflow.get_artifact_uri)


if __name__ == "__main__":
    main()
