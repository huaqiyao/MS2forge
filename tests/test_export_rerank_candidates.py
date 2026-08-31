import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_rerank_candidates.py"


def test_candidate_jsonl_to_label_free_tsv():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source = root / "candidate_cache.jsonl"
        output = root / "candidates.tsv"
        summary = root / "summary.json"

        molecule_record = {
            "schema": "flash_candidate_cache.v1",
            "type": "mol_candidates",
            "protocol": "diffms_2d",
            "rank": 0,
            "world_size": 1,
            "batch_idx": 7,
            "spec_id": "MassSpecGymID0000001",
            "true": {
                "input_smiles": "SHOULD_NOT_BE_EXPORTED",
                "canonical_smiles": "SHOULD_NOT_BE_EXPORTED",
            },
            "hits": {"1": True},
            "pred": {
                "n_generated": 10,
                "n_valid": 9,
                "n_valid_connected": 8,
                "n_invalid": 1,
                "n_disconnected": 1,
                "connected_smiles_nonisomeric_counts": [["CCO", 6], ["CCN", 2]],
            },
        }
        batch_marker = {
            "schema": "flash_candidate_cache.v1",
            "type": "batch_done",
            "rank": 0,
            "world_size": 1,
            "batch_idx": 7,
            "num_mols": 1,
        }
        source.write_text(
            json.dumps(molecule_record) + "\n" + json.dumps(batch_marker) + "\n",
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(source),
                "--output",
                str(output),
                "--summary-json",
                str(summary),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "ms2forge.rerank_candidate_table.v1" in completed.stdout

        with output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        assert [row["candidate_canonical_smiles"] for row in rows] == ["CCO", "CCN"]
        assert [row["candidate_rank_freq"] for row in rows] == ["1", "2"]
        assert [row["candidate_freq"] for row in rows] == ["6", "2"]
        assert [row["candidate_freq_fraction"] for row in rows] == ["0.6", "0.2"]
        assert rows[0]["candidate_hash"] == (
            "ms2forge_" + hashlib.sha256(b"CCO").hexdigest()[:24]
        )
        assert rows[0]["candidate_first_seen"] == ""
        assert rows[0]["rdkit_valid"] == "true"
        assert rows[0]["included_reason"] == "top_rank"
        assert "true" not in rows[0]
        assert "hits" not in rows[0]
        assert "SHOULD_NOT_BE_EXPORTED" not in output.read_text(encoding="utf-8")

        audit = json.loads(summary.read_text(encoding="utf-8"))
        assert audit["n_spectra"] == 1
        assert audit["n_candidates"] == 2
        assert audit["n_batch_markers"] == 1
        assert audit["truth_fields_exported"] == []
        assert audit["frozen_stage_a_ready"] is False
