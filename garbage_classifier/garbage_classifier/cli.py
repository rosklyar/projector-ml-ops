import typer
from garbage_data import load_train_data, load_data
from training_routine import train
from registry import upload_to_registry, download_from_registry
from inference import make_inference

app = typer.Typer()
app.command()(load_data)
app.command()(load_train_data)
app.command()(train)
app.command()(upload_to_registry)
app.command()(download_from_registry)
app.command()(make_inference)

if __name__ == "__main__":
    app()