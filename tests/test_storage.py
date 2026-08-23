from pathlib import Path

import pytest

from genframes.storage import STORAGE_ENVIRONMENT_VARIABLE, StorageLayout


def test_storage_requires_explicit_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(STORAGE_ENVIRONMENT_VARIABLE, raising=False)
    with pytest.raises(RuntimeError, match=STORAGE_ENVIRONMENT_VARIABLE):
        StorageLayout.resolve()


def test_storage_resolves_environment_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(STORAGE_ENVIRONMENT_VARIABLE, str(tmp_path))
    layout = StorageLayout.resolve()
    layout.create()
    assert layout.root == tmp_path.resolve()
    assert layout.datasets.is_dir()
    assert layout.checkpoints.is_dir()
    assert layout.benchmarks.is_dir()
