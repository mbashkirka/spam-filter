from __future__ import annotations

import argparse
from pathlib import Path

from spam_filter.config import ProjectPaths
from spam_filter.m8 import FrozenSpamService
from spam_filter.m9 import benchmark_service, write_latency_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--warmup', type=int, default=25)
    parser.add_argument('--slo-p95-ms', type=float, default=50.0)
    args = parser.parse_args()
    paths = ProjectPaths.from_root(args.root)
    service = FrozenSpamService.load(paths)
    result = benchmark_service(
        service,
        iterations=args.iterations,
        warmup=args.warmup,
        slo_p95_ms=args.slo_p95_ms,
    )
    write_latency_report(paths, result)
    latency = result['latency_ms']
    print(f"p50_ms={latency['p50']:.4f}")
    print(f"p95_ms={latency['p95']:.4f}")
    print(f"p99_ms={latency['p99']:.4f}")
    print(f"throughput_per_s={result['throughput_predictions_per_second']:.2f}")
    print(f"slo_passed={str(result['slo']['passed']).lower()}")
    print('milestone=M9')
    print('status=benchmarked')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
