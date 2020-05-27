import click

import extract_data


@click.command()
@click.option("--key", default='store47-2016.csv')
@click.option("--gcs_bucket", default='continuous-intelligence')
@click.option("--base_url", default='https://storage.googleapis.com')
@click.option("--build_number")
def main(key, gcs_bucket, base_url, build_number):
    extract_data.download_data(key, gcs_bucket, base_url, build_number)


if __name__ == "__main__":
    main()
