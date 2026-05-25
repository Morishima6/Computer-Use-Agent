import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

def _natural_sort_key(path: Path) -> Tuple:
    stem = path.stem
    match = re.match(r"^s(\d+)", stem, flags=re.IGNORECASE)
    if match:
        return (0, int(match.group(1)), path.name.lower(), path.name)
    return (1, stem.lower(), path.name)


def _iter_task_dirs(data_root: Path, task_id: str = "") -> Iterable[Path]:
    if task_id:
        target = data_root / task_id
        if not target.is_dir():
            raise FileNotFoundError(f"Task directory not found: {target}")
        yield target
        return

    for task_dir in sorted((p for p in data_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        yield task_dir


def _iter_screenshots(task_dir: Path) -> List[Path]:
    screenshots_dir = task_dir / "screenshots"
    if not screenshots_dir.is_dir():
        return []
    return sorted(screenshots_dir.glob("*.png"), key=_natural_sort_key)


def _build_engine_params(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "engine_type": args.provider,
        "model": args.model,
        "base_url": args.model_url,
        "api_key": args.model_api_key,
        "temperature": args.model_temperature,
    }


def _format_task_result_json(result: Dict[str, Dict[str, object]]) -> str:
    lines = ["{"]
    items = list(result.items())
    for index, (image_name, payload) in enumerate(items):
        trailing = "," if index < len(items) - 1 else ""
        summary = json.dumps(
            payload["step_retrieval_summary"],
            ensure_ascii=False,
        )
        embedding = json.dumps(
            payload["step_retrieval_summary_embeding"],
            ensure_ascii=False,
            separators=(", ", ": "),
        )
        lines.append(f'  {json.dumps(image_name, ensure_ascii=False)}: {{')
        lines.append(f'    "step_retrieval_summary": {summary},')
        lines.append(f'    "step_retrieval_summary_embeding": {embedding}')
        lines.append(f"  }}{trailing}")
    lines.append("}")
    return "\n".join(lines)


def _generate_task_result(
    task_dir: Path,
    engine_params: Dict[str, object],
    embedding_model: str,
    summarizer,
    generate_summary: Callable,
    get_embedding: Callable,
) -> Dict[str, Dict[str, object]]:
    result: Dict[str, Dict[str, object]] = {}

    for image_path in _iter_screenshots(task_dir):
        summary = generate_summary(
            image_path,
            engine_params=engine_params,
            summarizer=summarizer,
        )
        embedding = get_embedding(summary, model=embedding_model)
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError(
                f"Failed to generate embedding for {image_path.name} in task {task_dir.name}"
            )

        result[image_path.name] = {
            "step_retrieval_summary": summary,
            "step_retrieval_summary_embeding": embedding,
        }

    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate step_retrieval_summary and embedding files for _test screenshots."
    )
    parser.add_argument("--provider", type=str, required=True, help="LLM provider for summary generation.")
    parser.add_argument("--model", type=str, required=True, help="LLM model for summary generation.")
    parser.add_argument("--model_url", type=str, default="", help="Base URL for the summary model.")
    parser.add_argument("--model_api_key", type=str, default="", help="API key for the summary model.")
    parser.add_argument(
        "--model_temperature",
        type=float,
        default=None,
        help="Temperature for the summary model.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="text-embedding-v4",
        help="Embedding model passed to get_embedding.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(REPO_ROOT / "_test" / "data"),
        help="Directory containing _test task folders.",
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default=str(REPO_ROOT / "_test" / "result"),
        help="Directory where result/<task_id>/step_retrieval_summary.json will be written.",
    )
    parser.add_argument(
        "--task_id",
        type=str,
        default="",
        help="Optional single task id to process.",
    )
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    result_root = Path(args.result_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    from gui_agents.agents.step_retrieval_summary import (
        create_step_retrieval_summarizer,
        generate_step_retrieval_summary,
    )
    from trajectory.retrieval.common_llm_call import get_embedding

    engine_params = _build_engine_params(args)
    summarizer = create_step_retrieval_summarizer(engine_params)

    for task_dir in _iter_task_dirs(data_root, task_id=args.task_id):
        screenshots = _iter_screenshots(task_dir)
        print(f"[task] {task_dir.name}: {len(screenshots)} screenshots")
        task_result = _generate_task_result(
            task_dir=task_dir,
            engine_params=engine_params,
            embedding_model=str(args.embedding_model),
            summarizer=summarizer,
            generate_summary=generate_step_retrieval_summary,
            get_embedding=get_embedding,
        )
        out_dir = result_root / task_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "step_retrieval_summary.json"
        out_path.write_text(
            _format_task_result_json(task_result),
            encoding="utf-8",
        )
        print(f"[saved] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
