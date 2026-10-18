from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import check_no_dataset_text as chk

from evals.datasets import pag

SECRET = "the logging team diagnosed the root cause as monitoring gaps in the region"


def test_find_hits_plain_escaped_and_whitespace(tmp_path: Path) -> None:
    clean = tmp_path / "clean.md"
    clean.write_text("nothing to see here, only aggregate metrics 0.437")
    plain = tmp_path / "plain.txt"
    plain.write_text("x " + SECRET.upper().replace(" ", "\n  ", 3) + " y")  # case + line breaks
    escaped = tmp_path / "e.json"
    escaped.write_text(json.dumps({"a": SECRET + ' "quoted"'}))
    strings = {SECRET, chk._norm(SECRET + ' "quoted"')}
    hits = chk.find_hits([clean, plain, escaped], strings)
    assert str(clean) not in hits
    assert str(plain) in hits and str(escaped) in hits


def test_skips_binary_and_lock(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text(SECRET)
    blob = tmp_path / "b.bin"
    blob.write_bytes(b"\xff\xfe\x00" + SECRET.encode("utf-16"))
    assert chk.find_hits([lock, blob], {SECRET}) == {}


@pytest.mark.skipif(
    not pag.PAG_FILE.is_file(), reason="data/pag not fetched; the check is skipped too"
)
def test_repository_contains_no_pag_text() -> None:
    strings = chk.dataset_strings()
    assert len(strings) > 1000
    hits = chk.find_hits(chk.tracked_files(), strings)
    assert hits == {}, f"PAG dataset text committed in: {sorted(hits)}"


def test_main_skips_cleanly_without_data(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(chk, "PAG_FILE", tmp_path / "missing.json")
    assert chk.main() == 0
    assert "SKIPPED" in capsys.readouterr().out
