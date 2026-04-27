from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image as PILImage

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

try:
    from .screenshot_marker_utils import (
        DEFAULT_HIGHLIGHT_RADIUS,
        DEFAULT_HIGHLIGHT_WIDTH,
        DEFAULT_REGION_HEIGHT,
        DEFAULT_REGION_WIDTH,
        save_click_marked_artifacts,
    )
except ImportError:
    from screenshot_marker_utils import (
        DEFAULT_HIGHLIGHT_RADIUS,
        DEFAULT_HIGHLIGHT_WIDTH,
        DEFAULT_REGION_HEIGHT,
        DEFAULT_REGION_WIDTH,
        save_click_marked_artifacts,
    )


PROVIDER_CONFIGS = {
    "kimi": {
        "default_base_url": "https://api.chatanywhere.tech/v1",
        "default_model": "kimi-k2.5",
        "api_env_names": (
            "KIMI_API_KEY",
            "MOONSHOT_API_KEY",
            "OPENAI_API_KEY",
        ),
        "base_url_env_names": (
            "KIMI_BASE_URL",
            "MOONSHOT_BASE_URL",
            "OPENAI_BASE_URL",
        ),
    },
    "qwen": {
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-vl-plus",
        "api_env_names": (
            "QWEN_API_KEY",
            "DASHSCOPE_API_KEY",
            "OPENAI_API_KEY",
        ),
        "base_url_env_names": (
            "QWEN_BASE_URL",
            "OPENAI_BASE_URL",
        ),
    },
}

SUPPORTED_BATCH_ACTION_TYPES = {"click", "drag", "drag_to"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer the click position from a before-click screenshot with a configurable vision model, "
            "then generate the recorder-style marked image and (part) crop."
        )
    )
    parser.add_argument("input_image", help="Path to the original before-click screenshot.")
    parser.add_argument(
        "--output-dir",
        help="Directory to save outputs. Defaults to the input image directory.",
    )
    parser.add_argument(
        "--output-name",
        help="Marked image filename. Defaults to '<input_stem>_marked<input_suffix>'.",
    )
    parser.add_argument(
        "--part-name",
        help="Part image filename. Defaults to '<input_stem>(part)<suffix>'.",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDER_CONFIGS.keys()),
        default="kimi",
        help="Model provider preset. Controls the default API key/base URL/model. Default: kimi.",
    )
    parser.add_argument("--x", type=int, help="Manual x coordinate. Skips model inference when used with --y.")
    parser.add_argument("--y", type=int, help="Manual y coordinate. Skips model inference when used with --x.")
    parser.add_argument(
        "--hint",
        default="",
        help="Optional extra hint for the model, for example the intended UI element or task context.",
    )
    parser.add_argument("--api-key", default="", help="Provider API key. Overrides environment variables.")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL.")
    parser.add_argument(
        "--model",
        default="",
        help="Model name override. Defaults to the selected provider's default model.",
    )
    parser.add_argument(
        "--region-width",
        type=int,
        default=DEFAULT_REGION_WIDTH,
        help=f"Part crop width. Default: {DEFAULT_REGION_WIDTH}.",
    )
    parser.add_argument(
        "--region-height",
        type=int,
        default=DEFAULT_REGION_HEIGHT,
        help=f"Part crop height. Default: {DEFAULT_REGION_HEIGHT}.",
    )
    parser.add_argument(
        "--highlight-radius",
        type=int,
        default=DEFAULT_HIGHLIGHT_RADIUS,
        help=f"Red marker radius. Default: {DEFAULT_HIGHLIGHT_RADIUS}.",
    )
    parser.add_argument(
        "--highlight-width",
        type=int,
        default=DEFAULT_HIGHLIGHT_WIDTH,
        help=f"Red marker line width. Default: {DEFAULT_HIGHLIGHT_WIDTH}.",
    )
    return parser.parse_args()


def resolve_api_key(provider: str, explicit_api_key: str) -> str:
    if explicit_api_key:
        return explicit_api_key
    config = PROVIDER_CONFIGS[provider]
    for env_name in config["api_env_names"]:
        value = os.getenv(env_name)
        if value:
            return value
    raise RuntimeError(
        f"No API key found for provider '{provider}'. "
        f"Set one of: {', '.join(config['api_env_names'])}."
    )


def resolve_base_url(provider: str, explicit_base_url: str) -> str:
    if explicit_base_url:
        return explicit_base_url
    config = PROVIDER_CONFIGS[provider]
    for env_name in config["base_url_env_names"]:
        value = os.getenv(env_name)
        if value:
            return value
    return config["default_base_url"]


def resolve_model(provider: str, explicit_model: str) -> str:
    if explicit_model:
        return explicit_model
    return PROVIDER_CONFIGS[provider]["default_model"]


def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return "\n".join(part for part in text_parts if part)
    return str(content)


def parse_json_response(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find a JSON object in model output: {raw_text}")

    candidate = text[start : end + 1]

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    repaired = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', candidate)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    python_like = re.sub(r"\btrue\b", "True", repaired, flags=re.IGNORECASE)
    python_like = re.sub(r"\bfalse\b", "False", python_like, flags=re.IGNORECASE)
    python_like = re.sub(r"\bnull\b", "None", python_like, flags=re.IGNORECASE)

    try:
        parsed = ast.literal_eval(python_like)
        if isinstance(parsed, dict):
            return parsed
    except (SyntaxError, ValueError):
        pass

    raise ValueError(f"Failed to parse model output as JSON: {raw_text}")


def infer_click_position_with_model(
    image_path: Path,
    hint: str,
    api_key: str,
    base_url: str,
    model: str,
) -> tuple[int, int, dict[str, Any]]:
    with PILImage.open(image_path) as image:
        width, height = image.size

    prompt = (
        "You are given a GUI screenshot captured immediately before a click action.\n"
        "Infer the most likely click target coordinate.\n"
        f"The image size is width={width}, height={height}.\n"
        "Pick the exact point a user would most likely click, preferably near the center of the intended UI element.\n"
        "Return ONLY a raw JSON object with this schema:\n"
        "{\n"
        '  "x": integer,\n'
        '  "y": integer,\n'
        '  "confidence": number,\n'
        '  "reason": "short explanation"\n'
        "}\n"
        "Rules:\n"
        "- x must be within [0, width-1]\n"
        "- y must be within [0, height-1]\n"
        "- Do not return markdown\n"
        "- If uncertain, still provide the single best coordinate estimate\n"
    )
    if hint.strip():
        prompt += f"\nExtra hint:\n{hint.strip()}\n"

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise GUI grounding model. Output JSON only."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image(image_path)}"},
                    },
                ],
            },
        ],
        temperature=0.0,
    )

    raw_content = normalize_message_content(response.choices[0].message.content)
    parsed = parse_json_response(raw_content)
    if "x" not in parsed or "y" not in parsed:
        raise ValueError(f"Model output does not contain x/y: {raw_content}")

    x = max(0, min(int(parsed["x"]), width - 1))
    y = max(0, min(int(parsed["y"]), height - 1))
    return x, y, parsed


def resolve_output_paths(
    input_path: Path,
    output_dir_arg: str | None,
    output_name_arg: str | None,
    part_name_arg: str | None,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir_arg) if output_dir_arg else input_path.parent
    output_name = output_name_arg or f"{input_path.stem}_marked{input_path.suffix}"
    part_name = part_name_arg or f"{input_path.stem}(part){input_path.suffix}"
    return output_dir / output_name, output_dir / part_name


def locate_report_json(trajectory_dir: Path) -> Path:
    candidates = (
        trajectory_dir / "result" / "report.json",
        trajectory_dir / "results" / "report.json",
        trajectory_dir / "report.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find report.json under trajectory directory: {trajectory_dir}"
    )


def extract_position_from_step(step: dict[str, Any]) -> tuple[int, int] | None:
    action = step.get("action") or {}
    action_type = str(action.get("type") or "").strip().lower()
    target = action.get("target") or {}
    position = target.get("position")

    if isinstance(position, (list, tuple)) and len(position) >= 2:
        try:
            return int(position[0]), int(position[1])
        except (TypeError, ValueError):
            return None

    if action_type in {"drag", "drag_to"} and isinstance(position, dict):
        start = position.get("start")
        if isinstance(start, (list, tuple)) and len(start) >= 2:
            try:
                return int(start[0]), int(start[1])
            except (TypeError, ValueError):
                return None
    return None


def process_trajectory_directory(trajectory_dir: Path, args: argparse.Namespace) -> int:
    report_path = locate_report_json(trajectory_dir)
    report_data = json.loads(report_path.read_text(encoding="utf-8"))
    steps = report_data.get("steps") or []

    processed: list[dict[str, Any]] = []
    for step in steps:
        step_id = str(step.get("step_id", "")).strip() or "<unknown>"
        action = step.get("action") or {}
        action_type = str(action.get("type", "")).strip().lower()
        if action_type not in SUPPORTED_BATCH_ACTION_TYPES:
            processed.append(
                {
                    "step_id": step_id,
                    "action_type": action_type,
                    "status": "skip_unsupported_action_type",
                }
            )
            continue

        position = extract_position_from_step(step)
        if position is None:
            processed.append(
                {
                    "step_id": step_id,
                    "action_type": action_type,
                    "status": "skip_no_position",
                }
            )
            continue

        now_state = step.get("now_state") or {}
        before_rel = now_state.get("screenshot_path_before")
        if not before_rel:
            processed.append(
                {
                    "step_id": step_id,
                    "action_type": action_type,
                    "status": "skip_no_before_in_report",
                }
            )
            continue

        before_name = Path(str(before_rel)).name
        before_path = trajectory_dir / before_name
        marked_path = before_path.with_name(f"{before_path.stem}_marked{before_path.suffix}")
        part_path = before_path.with_name(f"{before_path.stem}(part){before_path.suffix}")

        if not before_path.exists():
            processed.append(
                {
                    "step_id": step_id,
                    "action_type": action_type,
                    "status": "skip_before_missing_on_disk",
                    "before_image": str(before_path),
                }
            )
            continue

        if marked_path.exists() or part_path.exists():
            processed.append(
                {
                    "step_id": step_id,
                    "action_type": action_type,
                    "status": "skip_marked_or_part_exists",
                    "before_image": str(before_path),
                    "marked_output": str(marked_path),
                    "part_output": str(part_path),
                }
            )
            continue

        saved_marked, saved_part = save_click_marked_artifacts(
            input_image_path=before_path,
            marked_output_path=marked_path,
            part_output_path=part_path,
            position=position,
            region_width=args.region_width,
            region_height=args.region_height,
            highlight_radius=args.highlight_radius,
            highlight_width=args.highlight_width,
        )
        processed.append(
            {
                "step_id": step_id,
                "action_type": action_type,
                "status": "processed",
                "before_image": str(before_path),
                "x": position[0],
                "y": position[1],
                "marked_output": str(saved_marked),
                "part_output": str(saved_part),
            }
        )

    summary = {
        "trajectory_dir": str(trajectory_dir),
        "report_json": str(report_path),
        "total_steps": len(steps),
        "processed_count": sum(1 for item in processed if item["status"] == "processed"),
        "skipped_count": sum(1 for item in processed if item["status"] != "processed"),
        "items": processed,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    load_dotenv()
    args = parse_args()

    input_path = Path(args.input_image).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    if input_path.is_dir():
        return process_trajectory_directory(input_path, args)

    marked_output_path, part_output_path = resolve_output_paths(
        input_path=input_path,
        output_dir_arg=args.output_dir,
        output_name_arg=args.output_name,
        part_name_arg=args.part_name,
    )

    if (args.x is None) != (args.y is None):
        raise ValueError("--x and --y must be provided together.")

    inference_payload: dict[str, Any]
    if args.x is not None and args.y is not None:
        x, y = args.x, args.y
        inference_payload = {"x": x, "y": y, "confidence": 1.0, "reason": "manual coordinates"}
    else:
        x, y, inference_payload = infer_click_position_with_model(
            image_path=input_path,
            hint=args.hint,
            api_key=resolve_api_key(args.provider, args.api_key),
            base_url=resolve_base_url(args.provider, args.base_url),
            model=resolve_model(args.provider, args.model),
        )

    marked_path, part_path = save_click_marked_artifacts(
        input_image_path=input_path,
        marked_output_path=marked_output_path,
        part_output_path=part_output_path,
        position=(x, y),
        region_width=args.region_width,
        region_height=args.region_height,
        highlight_radius=args.highlight_radius,
        highlight_width=args.highlight_width,
    )

    result = {
        "input_image": str(input_path),
        "x": x,
        "y": y,
        "marked_output": str(marked_path),
        "part_output": str(part_path),
        "model_output": inference_payload,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
