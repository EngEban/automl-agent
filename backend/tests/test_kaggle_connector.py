"""Unit tests for deferred Kaggle connector configuration."""
from __future__ import annotations

import builtins
import io
import types
import zipfile
from pathlib import Path

import pytest

from app.core.connectors.kaggle import KaggleConfigurationError, KaggleConnector


def _patch_kaggle_import(monkeypatch, kaggle_api_class):
    real_import = builtins.__import__
    fake_module = types.SimpleNamespace(KaggleApi=kaggle_api_class)

    def import_with_fake_kaggle(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "kaggle.api.kaggle_api_extended":
            return fake_module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_with_fake_kaggle)


def test_connector_reports_missing_credentials(monkeypatch) -> None:
    real_import = builtins.__import__

    def import_without_credentials(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "kaggle.api.kaggle_api_extended":
            raise OSError("Could not find kaggle.json")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_credentials)

    with pytest.raises(KaggleConfigurationError, match="Kaggle credentials"):
        KaggleConnector()


def test_connector_downloads_with_mocked_authenticated_api(monkeypatch, tmp_path: Path) -> None:
    class FakeKaggleApi:
        authenticated = False

        def authenticate(self) -> None:
            self.authenticated = True

        def dataset_download_files(self, reference, path, unzip) -> None:
            assert reference == "uciml/iris"
            assert path == str(tmp_path)
            assert unzip is False
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("iris.csv", "sepal_length,sepal_width\n5.1,3.5\n")
            (tmp_path / "iris.zip").write_bytes(archive.getvalue())

    _patch_kaggle_import(monkeypatch, FakeKaggleApi)

    connector = KaggleConnector()
    result = connector.download_to_staging("uciml/iris", tmp_path)

    assert connector.api.authenticated is True
    assert result["provider"] == "kaggle"
    assert result["source_id"] == "uciml/iris"
    assert Path(result["file_path"]).read_text() == (
        "sepal_length,sepal_width\n5.1,3.5\n"
    )
