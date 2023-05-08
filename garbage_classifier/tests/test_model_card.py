from pathlib import Path
from garbage_classifier import model_card as md

def test_score_model():
    result = md.create_model_card("test", "test", "test", "test", "test", "test", "test", "test")
    assert result.startswith("# test")

def test_save_model_card():
    md.save_model_card("test.md", "test")
    assert Path("test.md").exists()
    assert Path("test.md").read_text() == "test"
    Path("test.md").unlink()

