import os

import click

import data_reader as reader
import model
import tracking


@click.command()
@click.option("--data")
@click.option("--label_col")
@click.option("--max_depth", default=7)
@click.option("--n_trees", default=200)
@click.option("--lr", default=0.005)
@click.option("--mlflow_tracking_url", os.getenv('MLFLOW_TRACKING_URL'))
@click.option("--experiment_name", os.getenv('TENANT', 'local'))
@click.option("--build_number", os.getenv('BUILD_NUMBER', '0'))
def main(data, label_col, max_depth, n_trees, lr, mlflow_tracking_url, experiment_name, build_number):
    test_data, train_data = reader.load_data(data)

    pipeline_model = model.train_model(train_data, label_col, max_depth, n_trees, lr)

    rmse, mae, r2 = model.evaluate_model(pipeline_model, test_data, label_col)

    print("Model tree model (max_depth=%f, trees=%f, lr=%f):" % (max_depth, n_trees, lr))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    with tracking.TrackML(mlflow_tracking_url, experiment_name, build_number) as track:
        track.log_params(
            {
                "max_depth": max_depth,
                "n_trees": n_trees,
                "lr": lr
            })
        track.log_metrics(
            {
                "RMSE": rmse,
                "R2": r2,
                "MAE": mae
            })

        track.log_model("sklearn", pipeline_model, "retail_model")


if __name__ == "__main__":
    main()
