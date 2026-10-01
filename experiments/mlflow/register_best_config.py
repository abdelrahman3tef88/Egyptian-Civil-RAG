"""register_best_config.py - register the already-selected configuration.

This script is deliberately narrow. It does NOT:

  * re-run the experiments,
  * search the MLflow runs for the best configuration again,
  * regenerate the golden dataset, or
  * recompute faithfulness.

It ONLY:

  1. reads reports/mlflow/best_config.json (written by promote_best_config.py),
  2. packages that configuration as an MLflow model,
  3. registers it in the MLflow Model Registry (a new version),
  4. tags the version with its source run and settings, and
  5. assigns the "champion" alias to the new version.

Run with:  uv run python experiments/mlflow/register_best_config.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlflow_common import (  # noqa: E402
    configure_stdout,
    configure_tracking,
    ensure_experiment,
    load_experiment_config,
    read_json,
)


def parse_args(argv=None):
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Register the selected configuration in the MLflow Model Registry."
    )
    parser.add_argument(
        "--experiment-config",
        default=None,
        help="path to experiment_config.yaml (defaults to the one next to this script)",
    )
    parser.add_argument(
        "--best-config",
        default=None,
        help="path to best_config.json (defaults to reports/mlflow/best_config.json)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="registered model name (defaults to the configured name)",
    )
    parser.add_argument(
        "--alias",
        default=None,
        help="alias assigned to the new version (defaults to the configured alias)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate best_config.json and print the plan without registering",
    )
    return parser.parse_args(argv)


# Keys a best_config.json must carry to be registrable.
REQUIRED_KEYS = (
    "run_id",
    "embedding_model",
    "chunk_size",
    "chunk_overlap",
    "search_type",
    "top_k",
    "faithfulness",
    "golden_dataset_version",
)


def load_best_config(path):
    """Read and validate the stored selection."""
    if not path.exists():
        raise SystemExit(
            f"best_config.json not found: {path}\n"
            "Select a configuration first:\n"
            "  uv run python experiments/mlflow/promote_best_config.py"
        )

    payload = read_json(path)
    missing = [key for key in REQUIRED_KEYS if payload.get(key) is None]
    if missing:
        raise SystemExit(f"{path} is missing required keys: {missing}")
    return payload


def print_plan(best_config, model_name, alias):
    """Print exactly what will be registered."""
    print("CONFIGURATION TO REGISTER")
    print("-" * 78)
    for key in (
        "run_id",
        "embedding_model",
        "embedding_dimension",
        "chunk_size",
        "chunk_overlap",
        "search_type",
        "top_k",
        "faithfulness",
        "golden_dataset_version",
    ):
        print(f"  {key:<24} : {best_config.get(key)}")
    print("-" * 78)
    print(f"  registered model name    : {model_name}")
    print(f"  alias for the new version: {alias}")
    print("-" * 78)


def log_pyfunc_model(mlflow, python_model, artifact_dir, registered_model_name, tags):
    """Log the model, tolerating the MLflow 2.x / 3.x API difference.

    MLflow 3 renamed ``artifact_path`` to ``name``; the installed major
    version decides which keyword is used.
    """
    major = int(str(mlflow.__version__).split(".")[0])
    kwargs = {
        "python_model": python_model,
        "artifacts": {"config": str(artifact_dir)},
        "registered_model_name": registered_model_name,
        "tags": tags,
    }
    if major >= 3:
        kwargs["name"] = "config"
    else:
        kwargs["artifact_path"] = "config"
    return mlflow.pyfunc.log_model(**kwargs)


def main(argv=None):
    """Register the stored selection and alias it as champion."""
    args = parse_args(argv)
    configure_stdout()

    config = load_experiment_config(args.experiment_config)
    model_name = args.model_name or config["registry"]["model_name"]
    alias = args.alias or config["registry"]["champion_alias"]

    best_config_path = (
        Path(args.best_config)
        if args.best_config
        else Path(config["execution"]["reports_dir"]) / "best_config.json"
    )
    best_config = load_best_config(best_config_path)

    print(f"Reading selection   : {best_config_path}\n")
    print_plan(best_config, model_name, alias)

    if args.dry_run:
        print("--dry-run: best_config.json is valid; nothing was registered.")
        return 0

    # The model wrapper lives in the project package (src/registry).
    from registry.config_model import (
        CONFIG_FILENAME,
        build_config_model,
        config_tags,
    )

    import mlflow
    from mlflow.tracking import MlflowClient

    # The source run comes from best_config.json - it is NOT searched for.
    source_run_id = best_config.get("source_run_id") or best_config["run_id"]

    tracking_uri = configure_tracking(mlflow, config)
    experiment_name = config["experiment"]["name"]
    ensure_experiment(
        mlflow, experiment_name, config["experiment"].get("artifact_location")
    )

    print(f"MLflow tracking URI : {tracking_uri}")
    print(f"Registered model    : {model_name}")
    print(f"Alias               : {alias}\n")

    # Stage the configuration file that travels with the model.
    import tempfile

    tags = config_tags(best_config)
    # Link the registered version to the run that produced it.
    tags["source_run_id"] = str(source_run_id)

    with tempfile.TemporaryDirectory() as staging_dir:
        staging_path = Path(staging_dir) / CONFIG_FILENAME
        staging_path.write_text(
            Path(best_config_path).read_text(encoding="utf-8"), encoding="utf-8"
        )

        # A fresh run documents the registration itself and carries the
        # link back to the source run (finished runs are not written to).
        with mlflow.start_run(run_name=f"register-{model_name}") as run:
            mlflow.set_tags(
                {
                    "registration_for_run": str(source_run_id),
                    "registered_model_name": model_name,
                    "champion_alias": alias,
                    "experiment_phase": "chunking_x_embedding",
                }
            )
            model_info = log_pyfunc_model(
                mlflow,
                build_config_model(best_config),
                staging_dir,
                model_name,
                tags,
            )
            registration_run_id = run.info.run_id

    version = model_info.registered_model_version

    # Alias (MLflow 3 replaces the deprecated stages with aliases).
    client = MlflowClient()
    client.set_registered_model_alias(model_name, alias, version)

    # Version tags for traceability (set explicitly so they survive any
    # registry-side loss of the model-level tags).
    for key, value in tags.items():
        client.set_model_version_tag(model_name, version, key, value)
    client.set_model_version_tag(
        model_name, version, "registration_run_id", registration_run_id
    )
    client.update_model_version(
        name=model_name,
        version=version,
        description=(
            f"Selected by faithfulness={best_config.get('faithfulness')} in run "
            f"{source_run_id} (chunk_size={best_config.get('chunk_size')}, "
            f"chunk_overlap={best_config.get('chunk_overlap')}, "
            f"embedding_model={best_config.get('embedding_model')})."
        ),
    )

    print("REGISTRATION COMPLETE")
    print("-" * 78)
    print(f"  model name          : {model_name}")
    print(f"  version             : {version}")
    print(f"  alias               : {alias}")
    print(f"  source run id       : {source_run_id}")
    print(f"  registration run id : {registration_run_id}")
    print("-" * 78)
    print(
        "The model exposes the registered configuration:\n"
        f'  mlflow.pyfunc.load_model("models:/{model_name}@{alias}")'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


