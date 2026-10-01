"""promote_best_config.py - select the best configuration by Faithfulness.

This script ONLY selects:

    finished MLflow runs  ->  highest faithfulness  ->  best_config.json

It does not register anything (that is register_best_config.py) and it
does not touch configs/config.yaml. Copying the selected values into the
production configuration is a separate, explicitly approved step.

Run with:  uv run python experiments/mlflow/promote_best_config.py
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlflow_common import (  # noqa: E402
    configure_stdout,
    configure_tracking,
    load_experiment_config,
    write_json,
)

# Parameters that identify a run as belonging to this experiment phase.
PHASE_PARAMS = ("embedding_model", "chunk_size", "chunk_overlap")


def parse_args(argv=None):
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Select the best experiment configuration by faithfulness."
    )
    parser.add_argument(
        "--experiment-config",
        default=None,
        help="path to experiment_config.yaml (defaults to the one next to this script)",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="override the MLflow experiment name",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="metric used for the selection (defaults to the configured metric)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="where best_config.json is written",
    )
    return parser.parse_args(argv)


def _to_int(value):
    """Convert an MLflow string parameter to int when possible."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def collect_candidates(mlflow, experiment_name, metric):
    """Return the finished, phase-relevant runs ordered by the metric."""
    mlflow.set_experiment(experiment_name)
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f"metrics.{metric} DESC"],
        output_format="list",
    )

    candidates = []
    for run in runs:
        # Only completed runs with a real metric value can be promoted.
        if run.info.status != "FINISHED":
            continue

        score = run.data.metrics.get(metric)
        if score is None:
            continue

        params = run.data.params
        # A run without the phase parameters is not part of this study.
        if any(key not in params for key in PHASE_PARAMS):
            continue

        candidates.append(
            {
                "run_id": run.info.run_id,
                "source_run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName"),
                "embedding_model": params.get("embedding_model"),
                "embedding_dimension": _to_int(params.get("embedding_dimension")),
                "chunk_size": _to_int(params.get("chunk_size")),
                "chunk_overlap": _to_int(params.get("chunk_overlap")),
                "search_type": params.get("search_type"),
                "top_k": _to_int(params.get("top_k")),
                "faithfulness": float(score),
                "questions_evaluated": run.data.metrics.get("questions_evaluated"),
                "golden_dataset_version": params.get("golden_dataset_version"),
                "prompt_version": params.get("prompt_version"),
                # The generation model that answered, and the separate
                # model that judged faithfulness.
                "generation_model": params.get("generation_model"),
                "faithfulness_judge_model": params.get("faithfulness_judge_model"),

                # Which Faithfulness implementation produced the score.
                "faithfulness_evaluator": params.get("faithfulness_evaluator"),
                "faithfulness_backend": run.data.tags.get("faithfulness_backend"),
                "index_reused": run.data.tags.get("index_reused"),
            }
        )

    # Selection criterion: the highest faithfulness score, and nothing
    # else. Latency, cost, memory and model size are NOT part of this
    # ranking - they are deliberately excluded so the promoted
    # configuration is chosen on quality alone.
    candidates.sort(
        key=lambda candidate: (
            -candidate["faithfulness"],
            candidate["chunk_size"] if candidate["chunk_size"] is not None else 0,
            candidate["embedding_model"] or "",
        )
    )
    return candidates



def print_comparison(candidates, metric):
    """Print the ranked comparison of every candidate run."""
    print("=" * 92)
    print(f"RUN COMPARISON (metric: {metric})")
    print("=" * 92)
    print(
        f"{'#':<3} {'embedding_model':<42} {'size':>6} {'overlap':>8} "
        f"{metric:>13}  {'run_id':<10}"
    )
    print("-" * 92)
    for position, candidate in enumerate(candidates, start=1):
        print(
            f"{position:<3} {candidate['embedding_model']:<42} "
            f"{candidate['chunk_size']:>6} {candidate['chunk_overlap']:>8} "
            f"{candidate['faithfulness']:>13.4f}  {candidate['run_id'][:10]}"
        )
    print("=" * 92)


def main(argv=None):
    """Pick the best run and write best_config.json."""
    args = parse_args(argv)
    configure_stdout()

    config = load_experiment_config(args.experiment_config)
    metric = args.metric or config["evaluation"]["metric"]
    experiment_name = args.experiment_name or config["experiment"]["name"]
    reports_root = config["execution"]["reports_dir"]
    output_path = (
        Path(args.output) if args.output else Path(reports_root) / "best_config.json"
    )

    import mlflow

    tracking_uri = configure_tracking(mlflow, config)
    print(f"MLflow tracking URI : {tracking_uri}")
    print(f"MLflow experiment   : {experiment_name}")
    print(f"Selection metric    : {metric}\n")

    candidates = collect_candidates(mlflow, experiment_name, metric)
    if not candidates:
        raise SystemExit(
            f"No finished runs with a {metric!r} metric were found in "
            f"experiment {experiment_name!r}. Run the experiments first:\n"
            "  uv run python experiments/mlflow/run_experiments.py"
        )

    print_comparison(candidates, metric)

    best = candidates[0]
    payload = dict(best)
    payload.update(
        {
            "selection_metric": metric,
            "experiment_name": experiment_name,
            "candidates_considered": len(candidates),
            "selected_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    write_json(output_path, payload)

    print("\nSELECTED CONFIGURATION")
    print("-" * 92)
    for key in (
        "embedding_model",
        "embedding_dimension",
        "chunk_size",
        "chunk_overlap",
        "search_type",
        "top_k",
        "faithfulness",
        "golden_dataset_version",
        "run_id",
    ):
        print(f"  {key:<24} : {payload.get(key)}")
    print("-" * 92)
    print(f"Written to: {output_path}")
    print(
        "Next step (register the selection; no experiment is re-run):\n"
        "  uv run python experiments/mlflow/register_best_config.py"
    )
    print(
        "note: configs/config.yaml was NOT modified. Copy the selected values "
        "into the production configuration only after explicit approval."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

