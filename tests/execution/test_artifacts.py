"""Local artifact persistence, verification, and failure behavior."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import tensortrail as tt


def _session(
    *,
    model_version: str = "1.0",
    backend: str = "numpy",
    **overrides: object,
) -> tt.ExecutionSession:
    values: dict[str, object] = {
        "model_id": "classifier",
        "model_version": model_version,
        "framework": backend,
        "precision": "float64",
    }
    values.update(overrides)
    return tt.ExecutionSession.create(**values)  # type: ignore[arg-type]


def _completed_pair() -> tuple[tt.ExecutionSession, tt.ExecutionSession]:
    reference = _session(model_version="1.0")
    candidate = _session(model_version="2.0")
    reference.add_checkpoint(
        name="encoder.output",
        tensor=np.arange(6, dtype=np.float64).reshape(2, 3),
        stage_name="encoder",
    )
    reference.add_checkpoint(
        name="head.output",
        tensor=np.array([1.0, 2.0], dtype=np.float32),
        stage_name="head",
    )
    candidate.add_checkpoint(
        name="encoder.output",
        tensor=np.arange(6, dtype=np.float64).reshape(2, 3) + 0.25,
        stage_name="encoder",
    )
    candidate.add_checkpoint(
        name="head.output",
        tensor=np.array([1.0, 2.5], dtype=np.float32),
        stage_name="head",
    )
    reference.finalize()
    candidate.finalize()
    return reference, candidate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def test_execution_artifact_round_trip(tmp_path: Path) -> None:
    reference, _ = _completed_pair()
    storage = tt.FilesystemStorage(tmp_path / "artifacts")
    destination = storage.save_execution(reference)

    assert destination == storage.execution_path(reference.execution_id)
    assert (destination / "manifest.json").is_file()
    checkpoint = reference.checkpoints[0]
    assert (destination / "checkpoints" / checkpoint.checkpoint_id / "metadata.json").is_file()
    assert (destination / "checkpoints" / checkpoint.checkpoint_id / "tensor.npy").is_file()

    loaded = storage.load_execution(reference.execution_id)
    assert loaded.status == "completed"
    assert loaded.execution_id == reference.execution_id
    assert loaded.metadata == reference.metadata
    assert loaded.created_at == reference.created_at
    assert loaded.completed_at == reference.completed_at
    assert [item.checkpoint_name for item in loaded.checkpoints] == [
        "encoder.output",
        "head.output",
    ]
    for original, restored in zip(reference.checkpoints, loaded.checkpoints, strict=True):
        assert restored.tensor_reference == "artifact"
        assert restored.tensor_source == original.tensor_source
        np.testing.assert_array_equal(
            loaded.get_tensor(restored.checkpoint_id),
            reference.get_tensor(original.checkpoint_id),
        )


def test_array_like_source_is_preserved(tmp_path: Path) -> None:
    session = _session()
    session.add_checkpoint(name="values", tensor=[1, 2, 3])
    session.finalize()
    storage = tt.FilesystemStorage(tmp_path / "artifacts")
    storage.save_execution(session)
    loaded = storage.load_execution(session.execution_id)
    checkpoint = loaded.checkpoints[0]
    assert checkpoint.tensor_source == "array-like"
    assert checkpoint.tensor_reference == "artifact"
    assert loaded.get_tensor(checkpoint.checkpoint_id).tolist() == [1, 2, 3]


def test_failed_session_round_trip(tmp_path: Path) -> None:
    session = _session()
    session.add_checkpoint(name="partial", tensor=np.array([1.0]))
    session.fail("inference failed")
    storage = tt.FilesystemStorage(tmp_path / "artifacts")
    storage.save_execution(session)
    loaded = storage.load_execution(session.execution_id)
    assert loaded.status == "failed"
    assert loaded.failure_reason == "inference failed"
    assert loaded.completed_at is not None
    assert loaded.checkpoint_count == 1


def test_empty_execution_round_trip(tmp_path: Path) -> None:
    session = _session()
    session.finalize()
    storage = tt.FilesystemStorage(tmp_path / "artifacts")
    storage.save_execution(session)
    loaded = storage.load_execution(session.execution_id)
    assert loaded.status == "completed"
    assert loaded.checkpoints == ()
    assert loaded.checkpoint_count == 0


def test_execution_artifacts_are_never_overwritten(tmp_path: Path) -> None:
    session = _session()
    session.finalize()
    storage = tt.FilesystemStorage(tmp_path / "artifacts")
    storage.save_execution(session)
    with pytest.raises(tt.ArtifactError, match="never overwritten"):
        storage.save_execution(session)


@pytest.mark.parametrize("terminal", [False, True])
def test_save_requires_completed_or_failed_session(tmp_path: Path, terminal: bool) -> None:
    session = _session()
    if terminal:
        session.add_checkpoint(name="a", tensor=np.array([1.0]))
    storage = tt.FilesystemStorage(tmp_path / "artifacts")
    with pytest.raises(tt.ExecutionStateError, match="completed or failed"):
        storage.save_execution(session)


def test_validation_artifact_round_trip(tmp_path: Path) -> None:
    reference, candidate = _completed_pair()
    storage = tt.FilesystemStorage(tmp_path / "artifacts")
    storage.save_execution(reference)
    storage.save_execution(candidate)
    loaded_reference = storage.load_execution(reference.execution_id)
    loaded_candidate = storage.load_execution(candidate.execution_id)
    result = tt.validate(
        loaded_reference,
        loaded_candidate,
        policy=tt.ValidationPolicy(backend="numpy", atol=1e-9, rtol=1e-9),
    )
    destination = storage.save_validation(result)
    assert destination == storage.validation_path(result.validation_id)
    assert (destination / "manifest.json").is_file()
    assert (destination / "validation.json").is_file()

    loaded = storage.load_validation(result.validation_id)
    assert loaded.status == result.status
    assert loaded.validation_id == result.validation_id
    assert loaded.created_at == result.created_at
    assert loaded.summary == result.summary
    assert loaded.total_checkpoint_count == result.total_checkpoint_count
    assert loaded.checkpoint_results == result.checkpoint_results


def test_validation_artifacts_are_never_overwritten(tmp_path: Path) -> None:
    reference, candidate = _completed_pair()
    storage = tt.FilesystemStorage(tmp_path / "artifacts")
    result = tt.validate(
        reference,
        candidate,
        policy=tt.ValidationPolicy(backend="numpy"),
    )
    storage.save_validation(result)
    with pytest.raises(tt.ArtifactError, match="never overwritten"):
        storage.save_validation(result)


def test_manifest_records_checksums_and_sizes(tmp_path: Path) -> None:
    session = _session()
    array = np.arange(6, dtype=np.float64).reshape(2, 3)
    checkpoint = session.add_checkpoint(name="a", tensor=array)
    session.finalize()
    storage = tt.FilesystemStorage(tmp_path / "artifacts")
    destination = storage.save_execution(session)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["checkpoints"][0]
    assert entry["checkpoint"]["checkpoint_id"] == checkpoint.checkpoint_id
    assert entry["checkpoint"]["tensor_reference"] == "artifact"
    assert entry["tensor_nbytes"] == array.nbytes
    tensor_path = destination / entry["tensor_path"]
    metadata_path = destination / entry["metadata_path"]
    assert entry["tensor_sha256"] == _sha256(tensor_path)
    assert entry["metadata_sha256"] == _sha256(metadata_path)


def test_tampered_tensor_is_rejected(tmp_path: Path) -> None:
    session = _session()
    session.add_checkpoint(name="a", tensor=np.arange(4, dtype=np.float64))
    session.finalize()
    storage = tt.FilesystemStorage(tmp_path / "artifacts")
    destination = storage.save_execution(session)
    tensor_path = destination / "checkpoints" / session.checkpoints[0].checkpoint_id
    (tensor_path / "tensor.npy").write_bytes(b"tampered")
    with pytest.raises(tt.ArtifactIntegrityError, match="checksum"):
        storage.load_execution(session.execution_id)


def test_tampered_metadata_file_is_rejected(tmp_path: Path) -> None:
    session = _session()
    session.add_checkpoint(name="a", tensor=np.arange(4, dtype=np.float64))
    session.finalize()
    storage = tt.FilesystemStorage(tmp_path / "artifacts")
    destination = storage.save_execution(session)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["checkpoints"][0]
    metadata_path = destination / entry["metadata_path"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["checkpoint_name"] = "renamed"
    _write_json(metadata_path, metadata)
    entry["metadata_sha256"] = _sha256(metadata_path)
    _write_json(manifest_path, manifest)
    with pytest.raises(tt.ArtifactIntegrityError, match="does not match the manifest"):
        storage.load_execution(session.execution_id)


def test_tensor_shape_metadata_mismatch_is_rejected(tmp_path: Path) -> None:
    session = _session()
    session.add_checkpoint(name="a", tensor=np.arange(6, dtype=np.float64).reshape(2, 3))
    session.finalize()
    storage = tt.FilesystemStorage(tmp_path / "artifacts")
    destination = storage.save_execution(session)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["checkpoints"][0]
    entry["checkpoint"]["tensor_shape"] = [3, 2]
    metadata_path = destination / entry["metadata_path"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["tensor_shape"] = [3, 2]
    _write_json(metadata_path, metadata)
    entry["metadata_sha256"] = _sha256(metadata_path)
    _write_json(manifest_path, manifest)
    with pytest.raises(tt.ArtifactIntegrityError, match="payload shape"):
        storage.load_execution(session.execution_id)


def test_missing_and_malformed_artifacts_are_reported(tmp_path: Path) -> None:
    storage = tt.FilesystemStorage(tmp_path / "artifacts")
    with pytest.raises(tt.ArtifactError, match="does not exist"):
        storage.load_execution("exec_missing")

    session = _session()
    session.finalize()
    destination = storage.save_execution(session)
    (destination / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(tt.ArtifactIntegrityError, match="not valid JSON"):
        storage.load_execution(session.execution_id)


def test_missing_tensor_file_is_reported(tmp_path: Path) -> None:
    session = _session()
    session.add_checkpoint(name="a", tensor=np.arange(4, dtype=np.float64))
    session.finalize()
    storage = tt.FilesystemStorage(tmp_path / "artifacts")
    destination = storage.save_execution(session)
    tensor_file = destination / "checkpoints" / session.checkpoints[0].checkpoint_id / "tensor.npy"
    tensor_file.unlink()
    with pytest.raises(tt.ArtifactError, match="missing"):
        storage.load_execution(session.execution_id)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("schema_version", "9.9"),
        ("artifact_type", "validation"),
        ("execution_id", "exec_other"),
    ],
)
def test_manifest_identity_fields_are_verified(tmp_path: Path, field_name: str, value: str) -> None:
    session = _session()
    session.finalize()
    storage = tt.FilesystemStorage(tmp_path / "artifacts")
    destination = storage.save_execution(session)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field_name] = value
    _write_json(manifest_path, manifest)
    with pytest.raises(tt.ArtifactIntegrityError):
        storage.load_execution(session.execution_id)


def test_unsafe_artifact_ids_are_rejected(tmp_path: Path) -> None:
    storage = tt.FilesystemStorage(tmp_path / "artifacts")
    for identifier in ("../escape", "a/b", "_hidden"):
        with pytest.raises(tt.TensorValidationError):
            storage.execution_path(identifier)
        with pytest.raises(tt.TensorValidationError):
            storage.load_execution(identifier)
        with pytest.raises(tt.TensorValidationError):
            storage.validation_path(identifier)
        with pytest.raises(tt.TensorValidationError):
            storage.load_validation(identifier)


def test_tampered_validation_document_is_rejected(tmp_path: Path) -> None:
    reference, candidate = _completed_pair()
    storage = tt.FilesystemStorage(tmp_path / "artifacts")
    result = tt.validate(reference, candidate, policy=tt.ValidationPolicy(backend="numpy"))
    destination = storage.save_validation(result)
    document_path = destination / "validation.json"
    document_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(tt.ArtifactIntegrityError, match="checksum"):
        storage.load_validation(result.validation_id)


def test_missing_validation_artifact_is_reported(tmp_path: Path) -> None:
    storage = tt.FilesystemStorage(tmp_path / "artifacts")
    with pytest.raises(tt.ArtifactError, match="does not exist"):
        storage.load_validation("val_missing")
