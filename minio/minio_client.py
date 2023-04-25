from pathlib import Path
from minio import Minio


class MinioClientNative:
    
    def __init__(self, endpoint, access_key, secret_key, bucket_name: str) -> None:
        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)
        self.client = client
        self.bucket_name = bucket_name
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)

    def upload_file(self, key: str, file_path: Path, tags: dict = None):        
        self.client.fput_object(self.bucket_name, key, str(file_path), tags=tags)

    def download_file(self, key: str, file_path: Path):
        return self.client.fget_object(bucket_name=self.bucket_name, object_name=key, file_path=str(file_path))

    def file_exists(self, key: str):
        return self.client.stat_object(bucket_name=self.bucket_name, object_name=key)

    def delete_file(self, key: str):
        self.client.remove_object(bucket_name=self.bucket_name, object_name=key)