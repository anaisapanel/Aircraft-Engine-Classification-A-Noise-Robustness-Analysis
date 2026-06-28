"""
Data-integrity tests for the metadata layer.

These verify that the active pipeline's metadata flow is internally
consistent without requiring audio files:

  1. The shipped metadata CSVs have the columns each script expects.
  2. build_master_meta() runs and produces the schema train_and_evaluate
     downstream code reads from.
  3. make_noisy_meta() runs given a synthetic noisy CSV in the format that
     build_dataset_b_snr.py would write.
  4. get_audio_path() routes clean vs noisy rows to the right paths.

Audio I/O is never attempted; only the metadata DataFrames are exercised.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import data  # noqa: E402

REPO_ROOT     = Path(__file__).resolve().parent.parent
SAMPLE_META   = REPO_ROOT / "data" / "sample_meta_stratified_snr.csv"
AIRCRAFT_META = REPO_ROOT / "data" / "aircraft_meta_new.csv"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Shipped CSVs have the columns the active pipeline reads
# ──────────────────────────────────────────────────────────────────────────────

class TestShippedMetadataSchema:

    def test_sample_meta_csv_exists(self):
        assert SAMPLE_META.is_file(), f"missing: {SAMPLE_META}"

    def test_aircraft_meta_csv_exists(self):
        assert AIRCRAFT_META.is_file(), f"missing: {AIRCRAFT_META}"

    def test_sample_meta_columns_match_pipeline(self):
        """data.build_master_meta + get_audio_path read these columns."""
        df = pd.read_csv(SAMPLE_META)
        for col in ("filename", "fold", "class", "hex_id", "original_fold"):
            assert col in df.columns, f"sample_meta missing column: {col}"

    def test_aircraft_meta_has_engtype(self):
        """build_master_meta merges sample_meta with aircraft_meta on hex_id
        to bring in Engtype."""
        df = pd.read_csv(AIRCRAFT_META)
        assert "hex_id" in df.columns
        assert "Engtype" in df.columns

    def test_sample_meta_folds_are_three_stratified_folds(self):
        """src/data.py hardcodes FOLDS=[0,1,2].
        The shipped CSV must agree."""
        df = pd.read_csv(SAMPLE_META)
        assert sorted(df["fold"].unique()) == data.FOLDS

    def test_sample_meta_class_values_are_binary(self):
        df = pd.read_csv(SAMPLE_META)
        assert set(df["class"].unique()) <= {0, 1}


# ──────────────────────────────────────────────────────────────────────────────
# 2. build_master_meta() round-trip on the real CSVs
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildMasterMeta:

    @pytest.fixture(scope="class")
    def master(self):
        return data.build_master_meta(str(SAMPLE_META), str(AIRCRAFT_META))

    def test_returns_nonempty_dataframe(self, master):
        assert isinstance(master, pd.DataFrame)
        assert len(master) > 0

    def test_adds_engtype_flat_label_audio_root(self, master):
        for col in ("Engtype", "flat_label", "audio_root"):
            assert col in master.columns, f"missing: {col}"

    def test_background_engtype_is_background(self, master):
        bg = master[master["class"] == 0]
        assert (bg["Engtype"] == "background").all()

    def test_aircraft_engtype_is_one_of_subclasses(self, master):
        ac = master[master["class"] == 1]
        assert set(ac["Engtype"].unique()).issubset(set(data.SUBCLASSES))

    def test_no_diesel_engine_remains(self, master):
        assert (master["Engtype"] != "Diesel Engine").all()

    def test_no_aircraft_with_missing_engtype(self, master):
        ac = master[master["class"] == 1]
        assert ac["Engtype"].notna().all()

    def test_flat_label_equals_engtype(self, master):
        assert (master["flat_label"] == master["Engtype"]).all()

    def test_master_preserves_original_fold(self, master):
        """get_audio_path needs original_fold for clean rows."""
        assert "original_fold" in master.columns


# ──────────────────────────────────────────────────────────────────────────────
# 3. make_noisy_meta() against a synthetic B-meta CSV
# ──────────────────────────────────────────────────────────────────────────────

class TestMakeNoisyMeta:
    """
    build_dataset_b_snr.py writes one CSV per noise level. The schema is the
    input meta + 'snr_db' + 'source_noise_file', with 'filename' rewritten
    to 'fold_X/1/<name>.wav'. We synthesise such a CSV and feed it to
    make_noisy_meta() to confirm the contract.
    """

    @pytest.fixture
    def synthetic_b_meta(self, tmp_path):
        sample = pd.read_csv(SAMPLE_META)
        ac = sample[sample["class"] == 1].head(8).copy()
        ac["filename"]          = ac.apply(
            lambda r: f"fold_{int(r['fold'])}/1/{r['filename']}", axis=1
        )
        ac["snr_db"]            = 25.0
        ac["source_noise_file"] = "synthetic.wav"
        path = tmp_path / "B25_meta.csv"
        ac.to_csv(path, index=False)
        return path

    @pytest.fixture
    def clean_background(self):
        master = data.build_master_meta(str(SAMPLE_META), str(AIRCRAFT_META))
        bg = master[master["class"] == 0].copy()
        bg["audio_root"] = "/clean/audio/root"
        return bg

    def test_make_noisy_meta_concatenates_noisy_and_clean_bg(
        self, synthetic_b_meta, clean_background
    ):
        result = data.make_noisy_meta(
            str(synthetic_b_meta),
            str(AIRCRAFT_META),
            "/noisy/audio/root",
            clean_background,
        )
        n_noisy_aircraft = (result["class"] == 1).sum()
        n_clean_bg       = (result["class"] == 0).sum()
        assert n_noisy_aircraft == 8
        assert n_clean_bg == len(clean_background)

    def test_noisy_aircraft_get_noisy_audio_root(
        self, synthetic_b_meta, clean_background
    ):
        result = data.make_noisy_meta(
            str(synthetic_b_meta),
            str(AIRCRAFT_META),
            "/noisy/audio/root",
            clean_background,
        )
        ac = result[result["class"] == 1]
        assert (ac["audio_root"] == "/noisy/audio/root").all()

    def test_clean_background_keeps_its_audio_root(
        self, synthetic_b_meta, clean_background
    ):
        result = data.make_noisy_meta(
            str(synthetic_b_meta),
            str(AIRCRAFT_META),
            "/noisy/audio/root",
            clean_background,
        )
        bg = result[result["class"] == 0]
        assert (bg["audio_root"] == "/clean/audio/root").all()


# ──────────────────────────────────────────────────────────────────────────────
# 4. get_audio_path() routes clean vs noisy rows correctly
# ──────────────────────────────────────────────────────────────────────────────

class TestGetAudioPath:

    def test_noisy_filename_with_fold_prefix_uses_audio_root_directly(self):
        row = {
            "audio_root":    "/noisy/root",
            "filename":      "fold_1/1/abc.wav",
            "class":         1,
            "original_fold": 0,
        }
        p = data.get_audio_path(row)
        assert str(p) == "/noisy/root/fold_1/1/abc.wav"

    def test_clean_filename_uses_original_fold_for_path(self):
        row = {
            "audio_root":    "/clean/root",
            "filename":      "abc.wav",
            "class":         1,
            "original_fold": 4,
        }
        p = data.get_audio_path(row)
        assert str(p) == "/clean/root/fold_4/1/abc.wav"

    def test_clean_background_path_uses_class_zero_subdir(self):
        row = {
            "audio_root":    "/clean/root",
            "filename":      "bg.wav",
            "class":         0,
            "original_fold": 2,
        }
        p = data.get_audio_path(row)
        assert str(p) == "/clean/root/fold_2/0/bg.wav"


# ──────────────────────────────────────────────────────────────────────────────
# 5. Extracted audio (skipped if the WAV corpus is not present)
# ──────────────────────────────────────────────────────────────────────────────

AUDIO_ROOT = REPO_ROOT / "data" / "audio"
SNR_ROOT   = REPO_ROOT / "data" / "output_snr"


def _audio_extracted() -> bool:
    return AUDIO_ROOT.is_dir() and any(AUDIO_ROOT.glob("fold_*/*/*.wav"))


def _snr_extracted() -> bool:
    return SNR_ROOT.is_dir() and all(
        (SNR_ROOT / f"{n}_meta.csv").is_file() for n in ("B25", "B50", "B75")
    )


@pytest.mark.skipif(not _audio_extracted(),
                    reason="clean audio not extracted under data/audio/")
class TestExtractedCleanAudio:
    """Guard the sample_meta → WAV reference contract on disk."""

    @pytest.fixture(scope="class")
    def master_on_disk(self):
        m = data.build_master_meta(str(SAMPLE_META), str(AIRCRAFT_META))
        m["audio_root"] = str(AUDIO_ROOT)
        return m

    def test_every_sample_meta_row_resolves_to_existing_wav(self, master_on_disk):
        missing = [
            str(data.get_audio_path(r))
            for _, r in master_on_disk.iterrows()
            if not data.get_audio_path(r).is_file()
        ]
        assert not missing, (
            f"{len(missing)} metadata rows reference non-existent audio; "
            f"first: {missing[:3]}"
        )

    def test_every_referenced_wav_is_under_expected_fold_layout(self, master_on_disk):
        """Paths must be data/audio/fold_{0..4}/{0,1}/<filename>.wav."""
        for _, row in master_on_disk.head(200).iterrows():
            p = data.get_audio_path(row)
            assert p.parent.name in ("0", "1"), p
            assert p.parent.parent.name.startswith("fold_"), p
            assert p.parent.parent.parent == AUDIO_ROOT, p


@pytest.mark.skipif(not _snr_extracted(),
                    reason="SNR audio not extracted under data/output_snr/")
class TestExtractedNoisyAudio:
    """Guard the B-meta → WAV reference contract on disk."""

    @pytest.fixture(scope="class")
    def clean_bg(self):
        m = data.build_master_meta(str(SAMPLE_META), str(AIRCRAFT_META))
        m["audio_root"] = str(AUDIO_ROOT)
        return m[m["class"] == 0].copy()

    @pytest.mark.parametrize("name", ["B25", "B50", "B75"])
    def test_every_noisy_aircraft_row_resolves(self, clean_bg, name):
        meta_file = SNR_ROOT / f"{name}_meta.csv"
        b_root    = SNR_ROOT / name
        noisy     = data.make_noisy_meta(str(meta_file), str(AIRCRAFT_META),
                                        str(b_root), clean_bg)
        ac = noisy[noisy["class"] == 1]
        missing = [str(data.get_audio_path(r)) for _, r in ac.iterrows()
                   if not data.get_audio_path(r).is_file()]
        assert not missing, (
            f"{name}: {len(missing)}/{len(ac)} rows reference missing WAVs; "
            f"first: {missing[:3]}"
        )

    @pytest.mark.parametrize("name", ["B25", "B50", "B75"])
    def test_on_disk_wav_count_matches_aircraft_count(self, name):
        meta_file = SNR_ROOT / f"{name}_meta.csv"
        b_root    = SNR_ROOT / name
        n_rows = len(pd.read_csv(meta_file))
        n_wavs = sum(1 for _ in b_root.rglob("*.wav"))
        assert n_wavs == n_rows, f"{name}: {n_rows} rows vs {n_wavs} wavs"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
