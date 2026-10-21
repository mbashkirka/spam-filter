from __future__ import annotations

import argparse
import json
from pathlib import Path

from spam_filter import __version__
from spam_filter.config import ProjectPaths
from spam_filter.data import SplitConfig, load_manifest, prepare_uci_dataset
from spam_filter.m2 import run_m2
from spam_filter.m3 import run_m3
from spam_filter.m4 import M4Config, run_m4
from spam_filter.m5 import M5Config, run_m5
from spam_filter.m6 import M6Config, prepare_enron1_dataset, run_m6
from spam_filter.m7 import M7Config, run_m7
from spam_filter.m8 import build_m8_serving_artifact
from spam_filter.m9 import build_release_contract

PROJECT_NAME = "spam-filter-lab"
MILESTONE = "M9"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spam-filter")
    subparsers = parser.add_subparsers(dest="command")

    info_parser = subparsers.add_parser("info", help="Print project contract information.")
    info_parser.add_argument("--root", type=Path, default=Path.cwd())

    subparsers.add_parser("version", help="Print package version.")

    prepare_parser = subparsers.add_parser(
        "prepare-data", help="Fetch UCI SMS Spam Collection and create M1 splits/reports."
    )
    prepare_parser.add_argument("--root", type=Path, default=Path.cwd())
    prepare_parser.add_argument("--seed", type=int, default=42)

    data_info_parser = subparsers.add_parser("data-info", help="Print the M1 data manifest.")
    data_info_parser.add_argument("--root", type=Path, default=Path.cwd())

    train_parser = subparsers.add_parser(
        "train-m2", help="Run M2 preprocessing ablation and train scratch Naive Bayes."
    )
    train_parser.add_argument("--root", type=Path, default=Path.cwd())

    train_m3_parser = subparsers.add_parser(
        "train-m3", help="Benchmark sparse NLP models and freeze the M3 champion."
    )
    train_m3_parser.add_argument("--root", type=Path, default=Path.cwd())

    train_m4_parser = subparsers.add_parser(
        "train-m4", help="Calibrate the M3 champion and select a cost-sensitive threshold."
    )
    train_m4_parser.add_argument("--root", type=Path, default=Path.cwd())
    train_m4_parser.add_argument("--fp-cost", type=float, default=10.0)
    train_m4_parser.add_argument("--fn-cost", type=float, default=1.0)

    train_m5_parser = subparsers.add_parser(
        "train-m5", help="Fine-tune the M5 Transformer benchmark."
    )
    train_m5_parser.add_argument("--root", type=Path, default=Path.cwd())
    train_m5_parser.add_argument("--model-name", default="distilbert-base-uncased")
    train_m5_parser.add_argument("--epochs", type=int, default=3)
    train_m5_parser.add_argument("--batch-size", type=int, default=16)
    train_m5_parser.add_argument("--max-length", type=int, default=128)
    train_m5_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_m5_parser.add_argument(
        "--device", choices=["auto", "mps", "cuda", "cpu"], default="auto"
    )

    prepare_m6_parser = subparsers.add_parser(
        "prepare-m6", help="Download and prepare the external AUEB Enron1 benchmark."
    )
    prepare_m6_parser.add_argument("--root", type=Path, default=Path.cwd())

    eval_m6_parser = subparsers.add_parser(
        "eval-m6", help="Evaluate frozen M3/M4/M5 models zero-shot on Enron1."
    )
    eval_m6_parser.add_argument("--root", type=Path, default=Path.cwd())
    eval_m6_parser.add_argument("--batch-size", type=int, default=32)
    eval_m6_parser.add_argument("--max-length", type=int, default=128)
    eval_m6_parser.add_argument(
        "--device", choices=["auto", "mps", "cuda", "cpu"], default="auto"
    )

    analyze_m7_parser = subparsers.add_parser(
        "analyze-m7",
        help=(
            "Run external slices, error taxonomy, explainability, and robustness stress tests."
        ),
    )
    analyze_m7_parser.add_argument("--root", type=Path, default=Path.cwd())
    analyze_m7_parser.add_argument("--batch-size", type=int, default=32)
    analyze_m7_parser.add_argument("--max-length", type=int, default=128)
    analyze_m7_parser.add_argument(
        "--device", choices=["auto", "mps", "cuda", "cpu"], default="auto"
    )
    analyze_m7_parser.add_argument("--top-features", type=int, default=25)

    freeze_m8_parser = subparsers.add_parser(
        "freeze-m8", help="Freeze and fingerprint the M8 production serving artifact."
    )
    freeze_m8_parser.add_argument("--root", type=Path, default=Path.cwd())

    release_m9_parser = subparsers.add_parser(
        "release-m9", help="Validate and write the final M9 release contract."
    )
    release_m9_parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def _print_info(root: Path) -> None:
    paths = ProjectPaths.from_root(root)
    print(f"project={PROJECT_NAME}")
    print(f"version={__version__}")
    print(f"milestone={MILESTONE}")
    print(f"root={paths.root}")
    print("status=ready")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "info"):
        _print_info(getattr(args, "root", Path.cwd()))
        return 0
    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "prepare-data":
        paths = ProjectPaths.from_root(args.root)
        prepared = prepare_uci_dataset(paths, config=SplitConfig(seed=args.seed))
        print(f"manifest={prepared.manifest}")
        print(f"eda={prepared.eda_markdown}")
        print("status=prepared")
        return 0
    if args.command == "data-info":
        paths = ProjectPaths.from_root(args.root)
        print(json.dumps(load_manifest(paths), indent=2, sort_keys=True))
        return 0
    if args.command == "train-m2":
        result = run_m2(ProjectPaths.from_root(args.root))
        summary = {"selected": result["selected"], "test_metrics": result["test_metrics"]}
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "train-m3":
        result = run_m3(ProjectPaths.from_root(args.root))
        summary = {
            "selected_model": result["selected_model"],
            "selected_validation": result["selected_validation"],
            "test_metrics": result["test_metrics"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "train-m4":
        result = run_m4(
            ProjectPaths.from_root(args.root),
            M4Config(fp_cost=args.fp_cost, fn_cost=args.fn_cost),
        )
        summary = {
            "selected_model": result["selected_model"],
            "selected_threshold": result["selected_threshold"],
            "test_probability_metrics": result["test_probability_metrics"],
            "test_decision_metrics": result["test_decision_metrics"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "train-m5":
        result = run_m5(
            ProjectPaths.from_root(args.root),
            M5Config(
                model_name=args.model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_length=args.max_length,
                learning_rate=args.learning_rate,
                device=args.device,
            ),
        )
        summary = {
            "model_name": result["config"]["model_name"],
            "device": result["device"],
            "best_epoch": result["best_epoch"],
            "test_metrics": result["test_metrics"],
            "test_probability_metrics": result["test_probability_metrics"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "prepare-m6":
        result = prepare_enron1_dataset(ProjectPaths.from_root(args.root))
        print(f"dataset={result.csv_path}")
        print(f"rows_unique={result.rows_unique}")
        print("status=prepared")
        return 0
    if args.command == "eval-m6":
        result = run_m6(
            ProjectPaths.from_root(args.root),
            M6Config(
                batch_size=args.batch_size,
                max_length=args.max_length,
                device=args.device,
            ),
        )
        summary = {
            "dataset": result["dataset"],
            "models": result["models"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "analyze-m7":
        result = run_m7(
            ProjectPaths.from_root(args.root),
            M7Config(
                batch_size=args.batch_size,
                max_length=args.max_length,
                device=args.device,
                top_features=args.top_features,
            ),
        )
        summary = {
            "external_agreement": result["external"]["agreement"],
            "robustness": result["robustness"]["rows"],
            "device": result["robustness"]["device"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "freeze-m8":
        result = build_m8_serving_artifact(ProjectPaths.from_root(args.root))
        summary = {
            "model_version": result["model_version"],
            "threshold": result["serving_policy"]["threshold"],
            "artifact_fingerprint": result["artifact_fingerprint"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "release-m9":
        result = build_release_contract(ProjectPaths.from_root(args.root))
        summary = {
            "release_version": result["release_version"],
            "model_version": result["model_version"],
            "artifact_fingerprint": result["artifact_fingerprint"],
            "release_status": result["release_status"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
