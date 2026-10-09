"""Every test gets its own empty clone library: no test may touch data\voices."""
import pytest


@pytest.fixture(autouse=True)
def isolated_voice_library(tmp_path, monkeypatch):
    from app import voice_library
    folder = tmp_path / '_voices'
    folder.mkdir()
    monkeypatch.setattr(voice_library, 'LIBRARY_DIR', folder)
