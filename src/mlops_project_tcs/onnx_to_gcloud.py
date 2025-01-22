from google.cloud import storage
import typer
from typing import Annotated
from pathlib import Path
from google.oauth2 import service_account


app = typer.Typer()


@app.command()
def push(
    onnx_model_path: Annotated[
        str,
        typer.Option(
            "--onnx_model_path",
            "-m",
            help="Path to onnx model",
        ),
    ] = "model.onnx",
    bucket_name: Annotated[
        str, typer.Option("--repo_name", "-rn", help="Name of google cloud bucket")
    ] = "mlops_dtu_model_onnx",
    service_key_json: Annotated[
        str, typer.Option("--service_key", "-k", help="Service account key json with bucket write access")
    ] = "bucket_manager.json",
) -> None:
    """Pushes an onnx model to google cloud storage s3 bucket."""
    onnx_model_path = Path(onnx_model_path)
    blob_name = f"models/{onnx_model_path.name}"

    credentials = service_account.Credentials.from_service_account_file(service_key_json)
    client = storage.Client(credentials=credentials, project=credentials.project_id)
    bucket = client.get_bucket(bucket_name)

    blob = bucket.blob(blob_name)
    blob.upload_from_filename(onnx_model_path)


if __name__ == "__main__":
    app()
