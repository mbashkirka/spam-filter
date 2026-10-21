import json
from pathlib import Path

from spam_filter.config import ProjectPaths
from spam_filter.data import sha256_file
from spam_filter.m7 import verify_m7_outputs


def main() -> int:
    paths = ProjectPaths.from_root(Path.cwd())
    outputs = verify_m7_outputs(paths)
    if not all(outputs.values()):
        raise RuntimeError(f"Missing M7 outputs: {outputs}")
    payload = json.loads((paths.reports / "m7_results.json").read_text(encoding="utf-8"))
    for section, key in [
        ("external", "slice_metrics"),
        ("external", "error_audit"),
    ]:
        item = payload[section][key]
        path = paths.root / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Hash mismatch for {item['path']}")
    for key in ["robustness", "explainability"]:
        item = payload[key]
        path = paths.root / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Hash mismatch for {item['path']}")
    print(f"enron_rows={payload['external']['rows']}")
    print(f"both_wrong={payload['external']['agreement']['both_wrong']}")
    print(f"stress_spam_rows={payload['robustness']['spam_rows']}")
    print("artifact_hashes=ok")
    print("milestone=M7")
    print("status=ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
