import typer
from garbage_data import load_data
from trainer import train

app = typer.Typer()
app.command()(load_data)
app.command()(train)

if __name__ == "__main__":
    app()