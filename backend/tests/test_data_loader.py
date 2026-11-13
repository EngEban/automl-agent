from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.config import settings
from app.core import data_loader
from app.core.data_loader import DataLoadError, _clean_columns


def _write_csv(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8", newline="")
    return path


def test_cleaning_column_names_does_not_mutate_or_copy_the_data_values():
    original = pd.DataFrame({" feature name ": [1, 2, 3]})
    cleaned = _clean_columns(original)

    assert original.columns.tolist() == [" feature name "]
    assert cleaned.columns.tolist() == ["feature_name"]
    assert cleaned["feature_name"].tolist() == [1, 2, 3]
    assert np.shares_memory(
        cleaned["feature_name"].to_numpy(),
        original[" feature name "].to_numpy(),
    )


@pytest.mark.parametrize(
    "content, message",
    [("", "CSV"), ("id,value\n", "no data rows")],
)
def test_empty_and_header_only_csvs_fail_clearly(tmp_path, content, message):
    path = _write_csv(tmp_path / "empty.csv", content)

    with pytest.raises(DataLoadError, match=message):
        data_loader.load_from_path(path)


@pytest.mark.parametrize(
    "row_count, limit",
    [(4, 8), (8, 8)],
)
def test_csv_below_or_at_limit_keeps_every_row_and_is_not_marked_sampled(
    tmp_path, monkeypatch, row_count, limit
):
    path = _write_csv(
        tmp_path / "small.csv",
        "id,value\n" + "".join(f"{i},{i * 2}\n" for i in range(row_count)),
    )
    monkeypatch.setattr(settings, "MAX_ROWS_SAMPLE", limit)
    monkeypatch.setattr(settings, "CSV_CHUNK_SIZE", 3)

    frame = data_loader.load_from_path(path)

    assert frame.shape == (row_count, 2)
    assert frame.columns.tolist() == ["id", "value"]
    assert frame["id"].tolist() == list(range(row_count))
    assert frame.attrs["loading_metadata"] == {
        "original_row_count": row_count,
        "loaded_row_count": row_count,
        "sampling_applied": False,
        "sampling_limit": limit,
        "sampling_method": "none",
        "sampling_seed": settings.RANDOM_STATE,
        "chunk_size": 3,
    }
    assert data_loader.describe(frame)["loading_metadata"]["original_row_count"] == row_count


def test_csv_reservoir_sampling_is_uniform_seeded_ordered_and_bounded(
    tmp_path, monkeypatch
):
    path = _write_csv(
        tmp_path / "large.csv",
        "id,value\n" + "".join(f"{i},{i * 10}\n" for i in range(80)),
    )
    monkeypatch.setattr(settings, "MAX_ROWS_SAMPLE", 9)
    monkeypatch.setattr(settings, "CSV_CHUNK_SIZE", 7)
    monkeypatch.setattr(settings, "RANDOM_STATE", 19)

    first = data_loader.load_from_path(path)
    repeated = data_loader.load_from_path(path)
    assert first["id"].tolist() == repeated["id"].tolist()
    assert len(first) == 9
    assert first["id"].is_unique
    assert first["id"].tolist() == sorted(first["id"].tolist())
    assert first["id"].tolist() != list(range(9))
    assert first.attrs["loading_metadata"] == {
        "original_row_count": 80,
        "loaded_row_count": 9,
        "sampling_applied": True,
        "sampling_limit": 9,
        "sampling_method": "reservoir",
        "sampling_seed": 19,
        "chunk_size": 7,
    }

    monkeypatch.setattr(settings, "RANDOM_STATE", 20)
    other_seed = data_loader.load_from_path(path)
    assert other_seed["id"].tolist() != first["id"].tolist()


def test_csv_parsing_preserves_mixed_values_missingness_quotes_and_utf8(
    tmp_path, monkeypatch
):
    content = (
        "id,measure,category,note\n"
        '1,1.5,blue,"Paris, France"\n'
        '2,NA,red,"contains ""quotes"""\n'
        "3,three,blue,München\n"
        "4,4.5,NA,東京\n"
    )
    path = _write_csv(tmp_path / "mixed.csv", content)
    monkeypatch.setattr(settings, "MAX_ROWS_SAMPLE", 10)
    monkeypatch.setattr(settings, "CSV_CHUNK_SIZE", 2)

    frame = data_loader.load_from_path(path)

    assert frame.columns.tolist() == ["id", "measure", "category", "note"]
    assert frame["id"].tolist() == [1, 2, 3, 4]
    assert frame["measure"].dtype == object
    assert frame["measure"].tolist() == ["1.5", None, "three", "4.5"]
    assert frame["category"].tolist() == ["blue", "red", "blue", None]
    assert frame["note"].tolist() == [
        "Paris, France", 'contains "quotes"', "München", "東京"
    ]


def test_csv_keeps_source_nonempty_columns_even_if_sample_has_only_missing_values(
    tmp_path, monkeypatch
):
    path = _write_csv(
        tmp_path / "rare-value.csv",
        "id,rare\n1,NA\n2,NA\n3,NA\n4,9\n",
    )
    monkeypatch.setattr(settings, "MAX_ROWS_SAMPLE", 2)
    monkeypatch.setattr(settings, "CSV_CHUNK_SIZE", 2)

    class KeepInitialReservoir:
        def __init__(self, _seed):
            pass

        @staticmethod
        def randrange(stop):
            return stop - 1

    monkeypatch.setattr(data_loader.random, "Random", KeepInitialReservoir)

    frame = data_loader.load_from_path(path)

    assert frame["id"].tolist() == [1, 2]
    assert "rare" in frame.columns
    assert frame["rare"].isna().all()


@pytest.mark.parametrize(
    "setting, value",
    [
        ("MAX_ROWS_SAMPLE", 0),
        ("MAX_ROWS_SAMPLE", -2),
        ("CSV_CHUNK_SIZE", 0),
        ("CSV_CHUNK_SIZE", 1_000_001),
    ],
)
def test_invalid_sampling_settings_raise_configuration_errors(
    tmp_path, monkeypatch, setting, value
):
    path = _write_csv(tmp_path / "data.csv", "id\n1\n2\n")
    monkeypatch.setattr(settings, setting, value)

    with pytest.raises(DataLoadError, match=setting):
        data_loader.load_from_path(path)


def test_csv_reader_is_called_once_with_a_bounded_chunk_size(tmp_path, monkeypatch):
    path = _write_csv(
        tmp_path / "chunked.csv",
        "id\n" + "".join(f"{i}\n" for i in range(25)),
    )
    monkeypatch.setattr(settings, "MAX_ROWS_SAMPLE", 5)
    monkeypatch.setattr(settings, "CSV_CHUNK_SIZE", 4)
    original_read_csv = pd.read_csv
    calls = []

    def observed_read_csv(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(data_loader.pd, "read_csv", observed_read_csv)

    frame = data_loader.load_from_path(path)

    assert len(frame) == 5
    assert len(calls) == 1
    assert calls[0]["chunksize"] == 4


def test_completed_csv_chunks_are_released_before_the_next_chunk(monkeypatch):
    import gc
    import weakref

    class ChunkReader:
        def __init__(self):
            self.index = 0
            self.previous_chunk = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return self

        def __next__(self):
            gc.collect()
            if self.previous_chunk is not None:
                assert self.previous_chunk() is None
            if self.index == 3:
                raise StopIteration
            chunk = pd.DataFrame({"id": [self.index * 2, self.index * 2 + 1]})
            self.previous_chunk = weakref.ref(chunk)
            self.index += 1
            return chunk

    monkeypatch.setattr(data_loader.pd, "read_csv", lambda *_args, **_kwargs: ChunkReader())
    monkeypatch.setattr(settings, "MAX_ROWS_SAMPLE", 2)
    monkeypatch.setattr(settings, "CSV_CHUNK_SIZE", 2)

    frame = data_loader._read_csv_chunked(io.BytesIO(b"unused"))

    assert frame["id"].tolist() == sorted(frame["id"].tolist())
    assert frame.attrs["loading_metadata"]["original_row_count"] == 6


def test_tsv_path_uses_chunked_tab_delimited_parser(tmp_path, monkeypatch):
    path = _write_csv(tmp_path / "tabular.tsv", "id\tlabel\n1\tα\n2\tβ\n3\tγ\n")
    monkeypatch.setattr(settings, "MAX_ROWS_SAMPLE", 3)
    monkeypatch.setattr(settings, "CSV_CHUNK_SIZE", 2)

    frame = data_loader.load_from_path(path)

    assert frame.columns.tolist() == ["id", "label"]
    assert frame["label"].tolist() == ["α", "β", "γ"]
    assert frame.attrs["loading_metadata"]["chunk_size"] == 2


def test_uploaded_bytes_use_the_same_chunked_csv_parser(monkeypatch):
    monkeypatch.setattr(settings, "MAX_ROWS_SAMPLE", 3)
    monkeypatch.setattr(settings, "CSV_CHUNK_SIZE", 2)
    original_read_csv = pd.read_csv
    calls = []

    def observed_read_csv(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(data_loader.pd, "read_csv", observed_read_csv)

    frame = data_loader.load_from_bytes(
        b"id,label\n1,a\n2,b\n3,a\n4,b\n", "uploaded.csv"
    )

    assert len(frame) == 3
    assert calls[0]["chunksize"] == 2
    assert frame.attrs["loading_metadata"]["sampling_method"] == "reservoir"


def test_non_csv_formats_keep_full_reader_behavior_and_report_sample_scope(
    tmp_path, monkeypatch
):
    import json

    path = tmp_path / "records.json"
    path.write_text(
        json.dumps([{"id": i, "value": f"v-{i}"} for i in range(20)]),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "MAX_ROWS_SAMPLE", 5)

    frame = data_loader.load_from_path(path)

    assert len(frame) == 5
    assert frame["id"].tolist() == sorted(frame["id"].tolist())
    assert frame.attrs["loading_metadata"] == {
        "original_row_count": 20,
        "loaded_row_count": 5,
        "sampling_applied": True,
        "sampling_limit": 5,
        "sampling_method": "uniform_random_sample",
        "sampling_seed": settings.RANDOM_STATE,
        "chunk_size": None,
    }
    from app.core.report_generator import _normalize_data

    report_data = _normalize_data(data_loader.describe(frame))
    assert report_data["loading_metadata"]["original_row_count"] == 20
    assert report_data["loading_metadata"]["sampling_applied"] is True


def test_url_csv_download_spools_to_disk_then_uses_chunked_loader(
    monkeypatch,
):
    content = b"id,label\n1,a\n2,b\n3,a\n4,b\n"
    monkeypatch.setattr(data_loader, "_validate_remote_url", lambda _url: None)
    monkeypatch.setattr(settings, "MAX_ROWS_SAMPLE", 3)
    monkeypatch.setattr(settings, "CSV_CHUNK_SIZE", 2)

    class FakeResponse:
        status_code = 200
        headers = {"Content-Length": str(len(content))}

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def iter_content(chunk_size):
            assert chunk_size == data_loader.DOWNLOAD_CHUNK_SIZE
            yield content[:12]
            yield content[12:]

        @staticmethod
        def close():
            return None

    monkeypatch.setattr(data_loader.requests, "get", lambda *_args, **_kwargs: FakeResponse())

    frame = data_loader.load_from_url("https://example.test/synthetic.csv")

    assert len(frame) == 3
    assert frame.attrs["loading_metadata"]["original_row_count"] == 4
    assert frame.attrs["loading_metadata"]["chunk_size"] == 2


def test_upload_service_reads_temporary_csv_path_without_materializing_file_bytes(
    tmp_path, monkeypatch
):
    from app.services import session_service

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    temporary_upload = tmp_path / "upload.part"
    temporary_upload.write_text("id,value\n1,2\n2,4\n", encoding="utf-8")
    monkeypatch.setattr(settings, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(
        session_service,
        "_finish_create",
        lambda _db, **values: values,
    )

    result = session_service.create_session_from_upload(
        object(),
        user_id="user-1",
        name="synthetic",
        original_filename="synthetic.csv",
        temporary_path=temporary_upload,
    )

    assert result["df"]["id"].tolist() == [1, 2]
    assert result["df"].attrs["loading_metadata"]["original_row_count"] == 2
    assert Path(result["source_path"]).read_text(encoding="utf-8") == (
        "id,value\n1,2\n2,4\n"
    )


def test_loaded_sample_remains_compatible_with_target_and_preprocessing_workflows(
    tmp_path, monkeypatch
):
    from app.core.model_selector import detect_target_column
    from app.core.preprocessing import build_preprocessor

    path = _write_csv(
        tmp_path / "automl.csv",
        "feature,group,label\n" + "".join(
            f"{i},{'a' if i % 2 else 'b'},{i % 2}\n" for i in range(12)
        ),
    )
    monkeypatch.setattr(settings, "MAX_ROWS_SAMPLE", 12)
    monkeypatch.setattr(settings, "CSV_CHUNK_SIZE", 3)

    frame = data_loader.load_from_path(path)
    target = detect_target_column(frame, explicit_target="label")
    preprocessor, _ = build_preprocessor(frame.drop(columns=["label"]))
    features = preprocessor.fit_transform(frame.drop(columns=["label"]))

    assert target["status"] == "selected"
    assert features.shape[0] == len(frame)
    assert list(frame.columns) == ["feature", "group", "label"]
