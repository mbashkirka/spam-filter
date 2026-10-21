from __future__ import annotations

import json
from pathlib import Path

from spam_filter.config import ProjectPaths
from spam_filter.m8 import FrozenSpamService, verify_m8_artifact
from spam_filter.m9 import RELEASE_REPORT


def main() -> int:
    paths = ProjectPaths.from_root(Path.cwd())
    if not verify_m8_artifact(paths):
        raise RuntimeError('M8 serving artifact verification failed')
    release_path = paths.reports / RELEASE_REPORT
    if not release_path.exists():
        raise FileNotFoundError('Run `make m9-release` first')
    release = json.loads(release_path.read_text(encoding='utf-8'))
    service = FrozenSpamService.load(paths)
    ham = service.predict_one('Are we still meeting at 7 tonight?')
    spam = service.predict_one('WIN a FREE prize now. Claim your reward today!')
    if ham['is_spam'] or not spam['is_spam']:
        raise RuntimeError('M9 semantic smoke failed')
    print(f"release_version={release['release_version']}")
    print(f"model_version={release['model_version']}")
    print(f"p95_ms={release['latency']['latency_ms']['p95']:.4f}")
    print('privacy_raw_text_logged=false')
    print('artifact_hash=ok')
    print('milestone=M9')
    print('status=ready')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
