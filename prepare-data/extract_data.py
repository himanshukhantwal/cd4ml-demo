import os
import urllib.request


def download_data(key, gcs_bucket, base_url, build_number):
    data_dir = os.getenv('CD4ML_DATA_DIR', 'data')
    path = f"{data_dir}/raw"
    downloaded_file_path = os.path.join(path, build_number, key)

    if not os.path.exists(downloaded_file_path):
        url = "%s/%s/%s" % (base_url, gcs_bucket, key)
        os.makedirs(os.path.join(path, build_number), exist_ok=True)
        urllib.request.urlretrieve(url, downloaded_file_path)
