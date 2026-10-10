"""Path helpers work relative to the package, without writing source data."""

from pathlib import Path

import pytest

from ..utils import path


@pytest.fixture
def package_data(tmp_path, monkeypatch):
    package = tmp_path / "spectrome"
    data = package / "data"
    data.mkdir(parents=True)
    (data / "example.txt").write_text("test data")
    monkeypatch.setattr(path, "__file__", str(package / "utils" / "path.py"))
    return data


def test_get_file_path(package_data):
    actual = Path(path.get_file_path("example.txt"))
    assert actual == package_data / "example.txt"
    assert actual.read_text() == "test data"


def test_get_data_path(package_data):
    assert Path(path.get_data_path()) == package_data
