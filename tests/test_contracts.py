from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import tifffile

from puma_pipeline.config import PumaConfig
from puma_pipeline.constants import CANDIDATE_DTYPE, STAGE1_CANDIDATE_DTYPE, STAGE1_GT_FEATURE_DTYPE
from puma_pipeline.data.preprocess import CENTROID_DTYPE
from puma_pipeline.stage1.data import apply_dihedral
from puma_pipeline.stage1.trainer import _early_stopping_due
from puma_pipeline.stage2.biomask import BioMaskHead
from puma_pipeline.stage2.calibration import load_calibration
from puma_pipeline.stage2.cache import _detection_from_reference, _variant_xy
from puma_pipeline.stage2.dataset import Stage2Cache
from puma_pipeline.stage2.risk import RISK_DIM, risk_features
from puma_pipeline.stage2.spatial import build_knn_graph, spatial_features_from_reference
from puma_pipeline.stage2.tissue_context import tissue_context_from_probability_map
from puma_pipeline.store import PumaArtifactStore, load_stage1_oof, load_stage1_gt_features
from puma_pipeline.submission.grand_challenge import read_rgb
from puma_pipeline.submission.format import (
    FINAL_DTYPE,
    predictions_to_puma_polygons,
    validate_puma_nuclei_json,
    validate_tissue_mask,
    validate_tissue_tiff,
    write_puma_json,
    write_tissue_tiff,
)
from puma_pipeline.tissue.model import TissueHead
from puma_pipeline.utils.runtime import resolve_cuda_amp_dtype, runtime_batch_size


ROOT = Path(__file__).resolve().parents[1]


def test_training_and_deployment_fingerprints_match() -> None:
    train = PumaConfig.load(ROOT / "configs" / "train_config.json")
    deploy = PumaConfig.load(ROOT / "docker" / "deployment_config.json")
    assert train.stage1_fingerprint == deploy.stage1_fingerprint
    assert train.tissue_fingerprint == deploy.tissue_fingerprint
    assert train.fingerprint == deploy.fingerprint
    assert train.stage2_feature_contract == "self_exclusion_pixel_sampling"
    assert train.stage1_epochs == 30
    assert train.stage1_micro_batch_size == 8
    assert train.stage1_effective_batch_size == 8
    assert train.stage1_inference_batch_size == 8
    assert train.stage1_early_stopping_patience == 15
    assert train.stage2_epochs == 100
    assert train.stage2_output_dir == "PUMA_stage2_outputs"
    assert train.cache_dir == "PUMA_stage2_cache"
    assert not hasattr(train, "stage2_early_stopping_patience")



def test_stage1_early_stopping_uses_epoch_patience() -> None:
    assert not _early_stopping_due(epoch=19, best_epoch=5, patience=15, validation_ran=True)
    assert _early_stopping_due(epoch=20, best_epoch=5, patience=15, validation_ran=True)
    assert not _early_stopping_due(epoch=30, best_epoch=0, patience=15, validation_ran=True)
    assert not _early_stopping_due(epoch=30, best_epoch=5, patience=15, validation_ran=False)


def test_runtime_precision_falls_back_to_fp16_when_bf16_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert resolve_cuda_amp_dtype(True) == torch.float16
    assert resolve_cuda_amp_dtype(False) == torch.float16


def test_runtime_batch_cap_protects_16gb_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Props:
        total_memory = 16 * 1024**3

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: _Props())
    assert runtime_batch_size(8, training=True) == 1
    assert runtime_batch_size(8, training=False) == 1


def test_stage2_feature_contract_only_invalidates_full_fingerprint() -> None:
    cfg = PumaConfig.load(ROOT / "configs" / "train_config.json")
    stage1 = cfg.stage1_fingerprint
    tissue = cfg.tissue_fingerprint
    full = cfg.fingerprint
    cfg.stage2_feature_contract = "synthetic_changed_contract"
    assert cfg.stage1_fingerprint == stage1
    assert cfg.tissue_fingerprint == tissue
    assert cfg.fingerprint != full


def test_invalid_config_fails_fast() -> None:
    payload = json.loads((ROOT / "configs" / "train_config.json").read_text())
    payload["prefetch_factor"] = 0
    with pytest.raises(ValueError, match="prefetch_factor"):
        PumaConfig(**payload)
    payload = json.loads((ROOT / "configs" / "train_config.json").read_text())
    payload["label_smoothing"] = 1.0
    with pytest.raises(ValueError, match="label_smoothing"):
        PumaConfig(**payload)
    payload = json.loads((ROOT / "configs" / "train_config.json").read_text())
    payload["stage2_hidden_dim"] = 0
    with pytest.raises(ValueError, match="stage2_hidden_dim"):
        PumaConfig(**payload)
    payload = json.loads((ROOT / "configs" / "train_config.json").read_text())
    payload["stage2_views"] = ["V2", "V2"]
    with pytest.raises(ValueError, match="duplicates"):
        PumaConfig(**payload)


@pytest.mark.parametrize("code", range(8))
def test_dihedral_coordinate_matches_transformed_pixel_center(code: int) -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[2, 1] = 255
    coordinates = np.asarray([[1.5, 2.5]], dtype=np.float32)
    transformed, xy = apply_dihedral(image, coordinates, code)
    signal = transformed[..., 0]
    y, x = np.unravel_index(int(signal.argmax()), signal.shape)
    np.testing.assert_allclose(xy[0], [x + 0.5, y + 0.5], atol=1e-6)


def _candidate(x: float, y: float, score: float) -> np.ndarray:
    rows = np.zeros(1, dtype=CANDIDATE_DTYPE)
    rows["x"] = x
    rows["y"] = y
    rows["heatmap_score"] = score
    rows["quality"] = 0.8
    rows["uncertainty"] = 0.2
    rows["peak_sharpness"] = 0.3
    return rows


def test_spatial_detector_self_exclusion_differs_from_external_gt_query() -> None:
    ref = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float32)
    detector = spatial_features_from_reference(ref, ref, 1024, exclude_self=True)
    external = spatial_features_from_reference(ref[:1], ref, 1024, exclude_self=False)
    assert detector.shape == (3, 6)
    assert detector[0, 0] == pytest.approx(np.log1p(15.0), rel=1e-6)
    # The external GT point can see a detection at distance zero.
    assert external[0, 0] == pytest.approx(np.log1p(10.0), rel=1e-6)
    assert detector[0, 3] < external[0, 3]


def test_spatial_self_exclusion_rejects_misaligned_arrays() -> None:
    q = np.asarray([[0.0, 0.0]], np.float32)
    r = np.asarray([[1.0, 0.0]], np.float32)
    with pytest.raises(ValueError, match="identical"):
        spatial_features_from_reference(q, r, exclude_self=True)


def test_detection_self_exclusion_uses_identity_not_coordinate_guess() -> None:
    ref = np.concatenate((_candidate(5.5, 5.5, 0.2), _candidate(15.5, 5.5, 0.4)))
    detector = _detection_from_reference(ref, ref, 1024, exclude_self=True)
    external = _detection_from_reference(ref[:1], ref, 1024, exclude_self=False)
    # Detector rows exclude themselves when measuring nearest distance.
    assert detector[0, 5] == pytest.approx(np.log1p(10.0), rel=1e-6)
    assert external[0, 5] == pytest.approx(0.0, abs=1e-7)
    # Feature 7 stores the score margin.
    assert detector[0, 7] == pytest.approx(-0.2, abs=1e-6)
    assert external[0, 7] == pytest.approx(0.0, abs=1e-6)


def test_knn_graph_removes_self_by_index_with_duplicate_coordinates() -> None:
    xy = np.asarray([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0]], np.float32)
    index, distance = build_knn_graph(xy, k=2, radius_cap=100.0)
    assert index.shape == distance.shape == (3, 2)
    for row in range(3):
        assert row not in index[row].tolist()
    assert 1 in index[0].tolist()
    assert distance[0].min() == pytest.approx(0.0)


def test_tissue_context_uses_containing_pixel_not_bankers_rounding() -> None:
    prob = np.zeros((6, 4, 4), np.float32)
    for x in range(4):
        prob[x, :, x] = 1.0
    # Pixel center x=1.5 belongs to x-index 1.
    out = tissue_context_from_probability_map(
        prob, np.asarray([[1.5, 1.5]], np.float32), source_image_size=4, pool_radius=1
    )
    assert out.shape == (1, 18)
    assert out[0, 1] == pytest.approx(1.0)
    assert out[0, 2] == pytest.approx(0.0)


def test_risk_feature_validation_and_shape() -> None:
    prob = np.full((3, 10), 0.1, np.float32)
    one = np.ones(3, np.float32)
    result = risk_features(prob, one, one, one, one, prob.copy())
    assert result.shape == (3, RISK_DIM)
    with pytest.raises(ValueError, match=r"\[N,10\]"):
        risk_features(np.ones((3, 9), np.float32), one, one, one, one)
    with pytest.raises(ValueError, match="same number"):
        risk_features(prob, np.ones(2), one, one, one)


def test_groupnorm_models_accept_non_default_divisible_or_nondivisible_bases() -> None:
    # These channel sizes should work with GroupNorm.
    bio = BioMaskHead(base_channels=10)
    out = bio(torch.zeros((1, 3, 32, 32), dtype=torch.float32))
    assert out["mask_logits"].shape == (1, 1, 32, 32)
    tissue = TissueHead(in_channels=8, base=10, classes=6)
    logits = tissue(torch.zeros((1, 8, 8, 8), dtype=torch.float32), (16, 16))
    assert logits.shape == (1, 6, 16, 16)


def test_submission_rejects_negative_or_out_of_range_class_ids() -> None:
    rows = np.zeros(1, dtype=FINAL_DTYPE)
    rows["x"] = 10.5; rows["y"] = 20.5; rows["confidence"] = 0.8
    for bad in (-1, 10):
        rows["class_id"] = bad
        with pytest.raises(ValueError, match="invalid class_id"):
            predictions_to_puma_polygons(rows)


def test_submission_rejects_non_finite_geometry() -> None:
    rows = np.zeros(1, dtype=FINAL_DTYPE)
    rows["x"] = np.nan; rows["y"] = 20.5; rows["class_id"] = 0; rows["confidence"] = 0.8
    with pytest.raises(ValueError, match="non-finite coordinates"):
        predictions_to_puma_polygons(rows)


def test_submission_json_and_tissue_tiff_roundtrip(tmp_path: Path) -> None:
    rows = np.zeros(2, dtype=FINAL_DTYPE)
    rows["x"] = [10.5, 100.5]
    rows["y"] = [20.5, 200.5]
    rows["class_id"] = [0, 9]
    rows["confidence"] = [0.8, 1.2]
    json_path = tmp_path / "melanoma-10-class-nuclei-segmentation.json"
    write_puma_json(json_path, rows)
    payload = json.loads(json_path.read_text())
    validate_puma_nuclei_json(payload)
    assert payload["polygons"][1]["probability"] == pytest.approx(1.0)

    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[0, 0] = 5
    tissue_path = tmp_path / "images" / "melanoma-tissue-mask-segmentation" / "roi.tif"
    write_tissue_tiff(tissue_path, mask)
    restored = tifffile.imread(tissue_path)
    validate_tissue_mask(restored)
    validated = validate_tissue_tiff(tissue_path)
    assert np.array_equal(validated, mask)
    with tifffile.TiffFile(tissue_path) as tif:
        tags = tif.pages[0].tags
        assert tags["XResolution"].value == (300, 1)
        assert tags["YResolution"].value == (300, 1)
        assert tags["MinSampleValue"].value == 1
        assert tags["MaxSampleValue"].value == 5
    np.testing.assert_array_equal(restored, mask)


def test_submission_tissue_tiff_rejects_missing_required_metadata(tmp_path: Path) -> None:
    path = tmp_path / "bad.tif"
    tifffile.imwrite(path, np.zeros((1024, 1024), np.uint8), metadata=None)
    with pytest.raises(ValueError, match="missing required PUMA tags"):
        validate_tissue_tiff(path)


def _write_store(directory: Path, *, bad_offsets: bool = False, bad_class: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "puma_rgb_images.npy", np.zeros((1, 1024, 1024, 3), np.uint8), allow_pickle=False)
    np.save(directory / "puma_roi_manifest.npy", np.asarray([0], np.int32), allow_pickle=False)
    centroids = np.zeros(1, dtype=CENTROID_DTYPE)
    centroids["roi_index"] = 0; centroids["x"] = 1.5; centroids["y"] = 2.5; centroids["class_id"] = 10 if bad_class else 0
    np.save(directory / "puma_nuclei_centroids.npy", centroids, allow_pickle=False)
    offsets = np.asarray([0, 0 if bad_offsets else 1], np.int64)
    np.save(directory / "puma_roi_centroid_offsets.npy", offsets, allow_pickle=False)
    np.save(directory / "puma_fold_assignments.npy", np.asarray([0], np.int8), allow_pickle=False)


def test_artifact_store_validates_internal_contract(tmp_path: Path) -> None:
    valid = tmp_path / "valid"
    _write_store(valid)
    store = PumaArtifactStore.open(valid)
    assert len(store.roi_centroids(0)) == 1

    broken = tmp_path / "broken_offsets"
    _write_store(broken, bad_offsets=True)
    with pytest.raises(ValueError, match="offsets"):
        PumaArtifactStore.open(broken)

    broken_class = tmp_path / "broken_class"
    _write_store(broken_class, bad_class=True)
    with pytest.raises(ValueError, match="class_id"):
        PumaArtifactStore.open(broken_class)


def test_variant_xy_stays_inside_continuous_image_domain() -> None:
    rows = np.zeros(2, dtype=CANDIDATE_DTYPE)
    rows["x"] = [0.0, 1023.9999]
    rows["y"] = [0.0, 1023.9999]
    rows["gt_global_index"] = [0, -1]
    xy = _variant_xy(rows, variant=1, seed=123, image_size=1024)
    assert np.all(xy >= 0.0)
    assert np.all(xy < 1024.0)


def _write_stage2_cache(directory: Path, fingerprint: str = "fp") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    candidates = np.zeros(1, dtype=CANDIDATE_DTYPE)
    candidates["x"] = 1.5
    candidates["y"] = 2.5
    np.save(directory / "candidates.npy", candidates, allow_pickle=False)
    np.save(directory / "appearance.npy", np.zeros((1, 1, 1, 4608), np.float16), allow_pickle=False)
    np.save(directory / "biomask_rgb.npy", np.zeros((1, 1, 3, 16, 16), np.uint8), allow_pickle=False)
    np.save(directory / "biomask_targets_packed.npy", np.zeros((1, 1, 32), np.uint8), allow_pickle=False)
    np.save(directory / "spatial.npy", np.zeros((1, 6), np.float32), allow_pickle=False)
    np.save(directory / "detection.npy", np.zeros((1, 8), np.float32), allow_pickle=False)
    np.save(directory / "roi_relative_h.npy", np.zeros((1, 2), np.float32), allow_pickle=False)
    np.save(directory / "tissue.npy", np.zeros((1, 18), np.float32), allow_pickle=False)
    (directory / "cache_manifest.json").write_text(json.dumps({
        "artifact_schema": "puma",
        "config_fingerprint": fingerprint,
        "detection_dim": 8,
        "rows": 1,
        "variants": 1,
        "views": ["V2"],
    }))


def test_stage2_cache_contract_and_fingerprint_validation(tmp_path: Path) -> None:
    directory = tmp_path / "cache"
    _write_stage2_cache(directory, fingerprint="good")
    cache = Stage2Cache(directory, expected_fingerprint="good")
    batch = cache.batch(np.asarray([0], np.int64))
    assert batch["appearance"].shape == (1, 1, 4608)
    with pytest.raises(RuntimeError, match="fingerprint"):
        Stage2Cache(directory, expected_fingerprint="wrong")

    np.save(directory / "detection.npy", np.zeros((1, 9), np.float32), allow_pickle=False)
    with pytest.raises(RuntimeError, match="detection cache has shape"):
        Stage2Cache(directory, expected_fingerprint="good")


def test_risk_calibration_loader_validates_shape_finiteness_and_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "risk.npz"
    np.savez(path, weight=np.zeros((1, RISK_DIM), np.float32), bias=np.zeros(1, np.float32),
             threshold=np.float32(0.4), config_fingerprint=np.asarray(["abc"]))
    state, threshold = load_calibration(path, expected_fingerprint="abc")
    assert state["weight"].shape == (1, RISK_DIM)
    assert threshold == pytest.approx(0.4)
    with pytest.raises(RuntimeError, match="fingerprint"):
        load_calibration(path, expected_fingerprint="wrong")
    np.savez(path, weight=np.zeros((1, RISK_DIM - 1), np.float32), bias=np.zeros(1, np.float32), threshold=np.float32(0.4))
    with pytest.raises(RuntimeError, match="shape mismatch"):
        load_calibration(path)


def test_submission_rgb_reader_rejects_nonfinite_or_ambiguous_float_range(tmp_path: Path) -> None:
    bad_nan = tmp_path / "nan.tif"
    arr = np.zeros((4, 4, 3), np.float32); arr[0, 0, 0] = np.nan
    tifffile.imwrite(bad_nan, arr, photometric="rgb")
    with pytest.raises(ValueError, match="non-finite"):
        read_rgb(bad_nan)

    bad_range = tmp_path / "range.tif"
    tifffile.imwrite(bad_range, np.full((4, 4, 3), 1000.0, np.float32), photometric="rgb")
    with pytest.raises(ValueError, match="neither normalized"):
        read_rgb(bad_range)


def test_artifact_store_rejects_out_of_bounds_centroid(tmp_path: Path) -> None:
    directory = tmp_path / "bounds"
    _write_store(directory)
    centroids = np.load(directory / "puma_nuclei_centroids.npy", allow_pickle=False)
    centroids["x"] = 1024.0
    np.save(directory / "puma_nuclei_centroids.npy", centroids, allow_pickle=False)
    with pytest.raises(ValueError, match="inside the 1024x1024 ROI"):
        PumaArtifactStore.open(directory)


def test_stage1_cached_artifact_semantic_validation(tmp_path: Path) -> None:
    oof = np.zeros(1, dtype=STAGE1_CANDIDATE_DTYPE)
    oof["roi_index"] = 0; oof["candidate_index"] = 0; oof["x"] = 1.5; oof["y"] = 2.5
    oof["heatmap_score"] = 0.5; oof["quality"] = 0.5; oof["uncertainty"] = 0.2
    oof["peak_sharpness"] = 0.1; oof["nearest_distance"] = 10.0; oof["matched_gt_index"] = -1
    oof["match_distance"] = np.nan; oof["class_id"] = 10; oof["is_reject"] = 1; oof["fold"] = 0
    oof_path = tmp_path / "oof.npy"; np.save(oof_path, oof, allow_pickle=False)
    assert len(load_stage1_oof(oof_path, expected_folds=1)) == 1
    oof["class_id"] = 0
    np.save(oof_path, oof, allow_pickle=False)
    with pytest.raises(ValueError, match="inconsistent"):
        load_stage1_oof(oof_path, expected_folds=1)

    gt = np.zeros(1, dtype=STAGE1_GT_FEATURE_DTYPE)
    gt["source_id"] = -1; gt["roi_index"] = 0; gt["x"] = 1.5; gt["y"] = 2.5; gt["class_id"] = 0; gt["fold"] = 0
    gt["heatmap_score"] = 0.5; gt["quality"] = 0.5; gt["uncertainty"] = 0.2; gt["peak_sharpness"] = 0.1
    gt_path = tmp_path / "gt.npy"; np.save(gt_path, gt, allow_pickle=False)
    assert len(load_stage1_gt_features(gt_path, expected_folds=1)) == 1
    gt["x"] = 1024.0
    np.save(gt_path, gt, allow_pickle=False)
    with pytest.raises(ValueError, match="inside the 1024x1024 ROI"):
        load_stage1_gt_features(gt_path, expected_folds=1)
