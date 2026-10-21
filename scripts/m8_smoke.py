from pathlib import Path

from spam_filter.config import ProjectPaths
from spam_filter.m8 import FrozenSpamService, verify_m8_artifact, verify_m8_outputs


def main() -> int:
    paths = ProjectPaths.from_root(Path.cwd())
    if not verify_m8_artifact(paths):
        raise RuntimeError("M8 artifact verification failed")
    outputs = verify_m8_outputs(paths)
    if not all(outputs.values()):
        raise RuntimeError(f"Missing M8 outputs: {outputs}")
    service = FrozenSpamService.load(paths)
    ham = service.predict_one("Are we still meeting at 7 tonight?")
    spam = service.predict_one("WIN a FREE prize now. Claim your reward today!")
    if not 0.0 <= ham["spam_probability"] <= 1.0:
        raise RuntimeError("Ham probability is outside [0, 1]")
    if not 0.0 <= spam["spam_probability"] <= 1.0:
        raise RuntimeError("Spam probability is outside [0, 1]")
    if spam["spam_probability"] <= ham["spam_probability"]:
        raise RuntimeError("Smoke examples are not ordered by spam probability")
    print(f"model_version={service.manifest['model_version']}")
    print(f"threshold={service.model.threshold:.6f}")
    print(f"ham_probability={ham['spam_probability']:.4f}")
    print(f"spam_probability={spam['spam_probability']:.4f}")
    print("artifact_hash=ok")
    print("milestone=M8")
    print("status=ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
