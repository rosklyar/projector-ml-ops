from pathlib import Path
import pytest
from minio_client import MinioClientNative
from testcontainers.minio import MinioContainer
from minio.error import S3Error

@pytest.fixture(scope="session")
def minio_container():
    container = MinioContainer(image="quay.io/minio/minio:latest", access_key="test-access", secret_key="test-secret")
     # Spent 2 hours trying to figure out why the container hangs on Win 10 :)
    container.get_container_host_ip = lambda: 'localhost'
    container.start()
    yield container
    container.stop()


@pytest.fixture(scope="session")
def minio_client(minio_container: MinioContainer):
    minio_client = MinioClientNative(f'localhost:{minio_container.get_exposed_port(minio_container.port_to_expose)}','test-access','test-secret', 'test')
    yield minio_client

def test_upload_file(minio_client: MinioClientNative):
    minio_client.upload_file('test_upload_file', Path('minio/README.md'))
    assert minio_client.file_exists('test_upload_file')

def test_file_exists_throw_exception(minio_client: MinioClientNative):
    with pytest.raises(S3Error):
        minio_client.file_exists('non_existing_file')


def test_download_file(minio_client: MinioClientNative):
    minio_client.upload_file('test_download_file', Path('minio/README.md'))
    minio_client.download_file('test_download_file', Path('minio/README-downloaded.md'))
    assert Path('minio/README.md').read_text() == Path('minio/README-downloaded.md').read_text()
    Path('minio/README-downloaded.md').unlink()

def test_delete_file(minio_client: MinioClientNative):
    minio_client.upload_file('test_delete_file', Path('minio/README.md'))
    minio_client.delete_file('test_delete_file')
    with pytest.raises(S3Error):
        minio_client.file_exists('test_delete_file')