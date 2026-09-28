from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.rag.datasets import load_rag_cases
from evaluation.rag.runner import run_rag_experiment
from shopmind.app.core.config import load_app_config
from shopmind.app.main import _wire_runtime, create_app
from shopmind.app.rag.service import create_rag_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A3 RAG evaluation against indexed evidence")
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/a3"))
    parser.add_argument("--app-config", type=Path, default=Path("configs/app.yaml"))
    args = parser.parse_args()

    config = load_app_config(args.app_config)
    app = create_app(auto_wire=False)
    _wire_runtime(app, config=config)
    cases = load_rag_cases(args.cases)
    client = app.state.repository.client
    for path in args.configs:
        summary = run_rag_experiment(
            config_path=path,
            service_factory=lambda rag: create_rag_service(app.state.search_service, client, app.state.embedder, rag),
            cases=cases,
            runs_root=args.runs_root,
            base_config=config.rag,
        )
        print(f"{path}: {summary.run_dir} ({summary.successful_cases} succeeded, {summary.failed_cases} failed)")


if __name__ == "__main__":
    main()
