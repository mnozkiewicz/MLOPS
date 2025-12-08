import boto3
import os

def download_s3_folder(bucket_name: str, s3_prefix: str, local_dir: str):

    s3 = boto3.client("s3")

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        
        for obj in page["Contents"]:
            s3_key = obj["Key"]

            # Skip "folders"
            if s3_key.endswith("/"):
                continue

            local_path = os.path.join(local_dir, s3_key)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket_name, s3_key, local_path)



if __name__ == "__main__":
    BUCKET = "mlops-lab9-bucket"
    PREFIX = ""
    LOCAL_DIR = "src/model/saved_models"
    download_s3_folder(BUCKET, PREFIX, LOCAL_DIR)