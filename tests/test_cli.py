import json

from spam_filter.cli import main


def test_info_command(capsys, tmp_path) -> None:
    assert main(["info", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "project=spam-filter-lab" in out
    assert "milestone=M9" in out
    assert "status=ready" in out


def test_version_command(capsys) -> None:
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == "1.0.0"


def test_data_info_command(capsys, tmp_path) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "m1_manifest.json").write_text(
        json.dumps({"milestone": "M1", "rows_unique": 4}), encoding="utf-8"
    )
    assert main(["data-info", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert '"milestone": "M1"' in out
