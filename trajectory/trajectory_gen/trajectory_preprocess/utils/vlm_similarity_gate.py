from __future__ import annotations

import base64
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import httpx
from PIL import Image


DEFAULT_KIMI_BASE_URL = "https://api.kimi.com/coding/"
DEFAULT_KIMI_MODEL = "kimi-2.5"
DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
DEFAULT_ARK_MODEL ="Doubao-Seed-2.0-pro"
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen3.6-plus"
DEFAULT_GPT_BASE_URL = "https://api.chatanywhere.tech/v1"
DEFAULT_GPT_MODEL = "gpt-5.4"

LEGACY_KIMI_API_KEY = "sk-kimi-cgKbnKBcBVRKuefmW4eCnCE62zZjxklYhLkFoSVLIkcJtrPZBLLVHTfKVUeIR35U"
LEGACY_ARK_API_KEY = "cede0e54-37b6-40ab-87d0-e3002bb54031"
LEGACY_QWEN_API_KEY = "sk-4d920336c17e438f8c70e10c02f2ad83"
LEGACY_GPT_API_KEY = "sk-ZtYubzCGDuXOL12HQtyZnN4Lp0cDyw25DFWk4eOK7AS88Nzw"

system_prompt = """You are an GUI action judgement. Your task is to judge whether the action is accidental, redundant or performed aimlessly based the given screenshots.
You will be given:
    - action type
    - action before screenshot: the state of screen before this action
    - action after screenshot: the state of screen after 1s of this action
    - the next action before screenshot: the state of screen before the next action (maybe after 2-10s of the current action)

Note:
    - If there is a red marker in the before screenshot, it is only used to indicate the mouse position; please IGNORE the red marker.
    - Do not judge only by similarity. You know, sometimes selecting a toolbar button may produce similar before and after frames, but this does not mean that the action is aimless. At the same time, we must not overlook genuine redundancy, such as clicking in a blank space or an invalid input.
    - Not only look at the before and after screenshots of this action, pay attention to the next action before screenshot. Because sometimes an action doesn't manifest within 1 second, but may change over 2-10 seconds.

Output Formate (JSON):
{
    "Judgment": "yes or no",
    "Reason": "short explanation"
}

Output Explanation:
    - Say "yes" at "Judgment" when this action is accidental, redundant, failed or performed aimlessly.
    - Say "no" at "Judgment" when this action is meaningful, sucessful or purposeful

IMPORTANT: Your output must be in ENGLISH.
"""


def build_vlm_http_client(*, trust_env: bool = False):
    timeout = httpx.Timeout(connect=20.0, read=300.0, write=60.0, pool=180.0)
    return httpx.Client(
        http2=False,
        timeout=timeout,
        transport=httpx.HTTPTransport(retries=3, trust_env=trust_env),
        trust_env=trust_env,
    )


def extract_json_object(text: str):
    stripped = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        stripped = fence_match.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find JSON object in verifier response: {text}")
    return json.loads(stripped[start : end + 1])


def encode_image(path: Path, *, image_url: bool = False, compact: bool = False) -> Dict[str, Any]:
    if compact:
        with Image.open(path) as img:
            image = img.convert("RGB")
            image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=80, optimize=True)
        media_type = "image/jpeg"
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
    else:
        media_type = "image/png"
        data = base64.b64encode(path.read_bytes()).decode("ascii")

    if image_url:
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{data}"},
        }
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def resolve_backend_candidates(backend: str) -> List[str]:
    if backend != "auto":
        return [backend]
    return ["kimi", "ark", "qwen", "gpt"]


def _resolve_backend_config(
    backend: str,
    *,
    kimi_api_key: Optional[str],
    kimi_base_url: str,
    kimi_model: str,
    ark_api_key: Optional[str],
    ark_base_url: str,
    ark_model: str,
    qwen_api_key: Optional[str],
    qwen_base_url: str,
    qwen_model: str,
    gpt_api_key: Optional[str],
    gpt_base_url: str,
    gpt_model: str,
) -> Tuple[str, str, str]:
    if backend == "kimi":
        api_key = (
            kimi_api_key
            or os.environ.get("KIMI_API_KEY")
            or os.environ.get("KIMI_CODING_API_KEY")
            or LEGACY_KIMI_API_KEY
            or ""
        ).strip()
        base_url = (kimi_base_url or os.environ.get("KIMI_BASE_URL") or DEFAULT_KIMI_BASE_URL).strip()
        model = (kimi_model or os.environ.get("KIMI_MODEL") or DEFAULT_KIMI_MODEL).strip()
    elif backend == "ark":
        api_key = (
            ark_api_key
            or os.environ.get("ARK_API_KEY")
            or os.environ.get("ARK_CODING_API_KEY")
            or LEGACY_ARK_API_KEY
            or ""
        ).strip()
        base_url = (ark_base_url or os.environ.get("ARK_BASE_URL") or DEFAULT_ARK_BASE_URL).strip()
        model = (ark_model or os.environ.get("ARK_MODEL") or DEFAULT_ARK_MODEL).strip()
    elif backend == "qwen":
        api_key = (
            qwen_api_key
            or os.environ.get("QWEN_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or LEGACY_QWEN_API_KEY
            or ""
        ).strip()
        base_url = (
            qwen_base_url
            or os.environ.get("QWEN_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_QWEN_BASE_URL
        ).strip()
        model = (qwen_model or os.environ.get("QWEN_MODEL") or DEFAULT_QWEN_MODEL).strip()
    elif backend == "gpt":
        api_key = (
            gpt_api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("GPT_API_KEY")
            or LEGACY_GPT_API_KEY
            or ""
        ).strip()
        base_url = (
            gpt_base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("GPT_BASE_URL")
            or DEFAULT_GPT_BASE_URL
        ).strip()
        model = (
            gpt_model
            or os.environ.get("OPENAI_MODEL")
            or os.environ.get("GPT_MODEL")
            or DEFAULT_GPT_MODEL
        ).strip()
    else:
        raise RuntimeError(f"Unsupported backend: {backend}")

    if not api_key:
        raise RuntimeError(f"Missing API key for backend {backend}.")
    return api_key, base_url, model


def _normalize_screenshots(screenshots: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in screenshots:
        path = Path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Screenshot not found: {path}")
        normalized.append(
            {
                "path": path,
                "label": str(item.get("label", "")).strip(),
            }
        )
    return normalized


def _build_multimodal_user_content(
    prompt_text: str,
    screenshots: Sequence[Dict[str, Any]],
    *,
    image_url: bool,
    compact_images: bool,
    include_system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if include_system_prompt:
        content.append({"type": "text", "text": include_system_prompt})
    if prompt_text.strip():
        content.append({"type": "text", "text": prompt_text})
    for item in screenshots:
        if item["label"]:
            content.append({"type": "text", "text": item["label"]})
        content.append(encode_image(item["path"], image_url=image_url, compact=compact_images))
    return content


def _call_anthropic_json(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt_text: str,
    prompt_text: str,
    screenshots: Sequence[Dict[str, Any]],
    max_tokens: int,
    trust_env: bool,
    compact_images: bool,
) -> Dict[str, Any]:
    import anthropic

    client = anthropic.Anthropic(
        base_url=base_url,
        api_key=api_key,
        http_client=build_vlm_http_client(trust_env=trust_env),
    )
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt_text,
        messages=[
            {
                "role": "user",
                "content": _build_multimodal_user_content(
                    prompt_text,
                    screenshots,
                    image_url=False,
                    compact_images=compact_images,
                ),
            }
        ],
    )
    text = "\n".join(block.text for block in response.content if getattr(block, "type", "") == "text").strip()
    return extract_json_object(text)


def _call_openai_json(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt_text: str,
    prompt_text: str,
    screenshots: Sequence[Dict[str, Any]],
    max_tokens: int,
    trust_env: bool,
    compact_images: bool,
    use_system_message: bool,
    request_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=build_vlm_http_client(trust_env=trust_env),
    )
    messages: List[Dict[str, Any]]
    if use_system_message:
        messages = [
            {"role": "system", "content": system_prompt_text},
            {
                "role": "user",
                "content": _build_multimodal_user_content(
                    prompt_text,
                    screenshots,
                    image_url=True,
                    compact_images=compact_images,
                ),
            },
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": _build_multimodal_user_content(
                    prompt_text,
                    screenshots,
                    image_url=True,
                    compact_images=compact_images,
                    include_system_prompt=system_prompt_text,
                ),
            }
        ]

    create_kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if request_kwargs:
        create_kwargs.update(request_kwargs)
    response = client.chat.completions.create(**create_kwargs)
    return extract_json_object(response.choices[0].message.content or "")


def judge_multimodal_json_with_meta(
    *,
    system_prompt_text: str,
    prompt_text: str,
    screenshots: Sequence[Dict[str, Any]],
    backend: str = "kimi",
    http_trust_env: bool = False,
    max_tokens: int = 1024,
    compact_images: bool = True,
    openai_use_system_message: bool = True,
    provider_request_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    kimi_api_key: Optional[str] = None,
    kimi_base_url: str = DEFAULT_KIMI_BASE_URL,
    kimi_model: str = DEFAULT_KIMI_MODEL,
    ark_api_key: Optional[str] = None,
    ark_base_url: str = DEFAULT_ARK_BASE_URL,
    ark_model: str = DEFAULT_ARK_MODEL,
    qwen_api_key: Optional[str] = None,
    qwen_base_url: str = DEFAULT_QWEN_BASE_URL,
    qwen_model: str = DEFAULT_QWEN_MODEL,
    gpt_api_key: Optional[str] = None,
    gpt_base_url: str = DEFAULT_GPT_BASE_URL,
    gpt_model: str = DEFAULT_GPT_MODEL,
) -> Tuple[Dict[str, Any], str, str]:
    normalized_screenshots = _normalize_screenshots(screenshots)
    provider_request_kwargs = provider_request_kwargs or {}
    last_error: Optional[Exception] = None

    for provider in resolve_backend_candidates(backend):
        try:
            api_key, base_url, model = _resolve_backend_config(
                provider,
                kimi_api_key=kimi_api_key,
                kimi_base_url=kimi_base_url,
                kimi_model=kimi_model,
                ark_api_key=ark_api_key,
                ark_base_url=ark_base_url,
                ark_model=ark_model,
                qwen_api_key=qwen_api_key,
                qwen_base_url=qwen_base_url,
                qwen_model=qwen_model,
                gpt_api_key=gpt_api_key,
                gpt_base_url=gpt_base_url,
                gpt_model=gpt_model,
            )
            if provider == "kimi":
                parsed = _call_anthropic_json(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    system_prompt_text=system_prompt_text,
                    prompt_text=prompt_text,
                    screenshots=normalized_screenshots,
                    max_tokens=max_tokens,
                    trust_env=http_trust_env,
                    compact_images=compact_images,
                )
            else:
                parsed = _call_openai_json(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    system_prompt_text=system_prompt_text,
                    prompt_text=prompt_text,
                    screenshots=normalized_screenshots,
                    max_tokens=max_tokens,
                    trust_env=http_trust_env,
                    compact_images=compact_images,
                    use_system_message=openai_use_system_message,
                    request_kwargs=provider_request_kwargs.get(provider),
                )
            return parsed, provider, model
        except Exception as exc:
            last_error = exc
            if backend != "auto":
                raise

    raise RuntimeError(f"All VLM backends failed: {last_error}")


def judge_multimodal_json(**kwargs: Any) -> Dict[str, Any]:
    parsed, _, _ = judge_multimodal_json_with_meta(**kwargs)
    return parsed


def judge_action(
    action_type: str,
    before_path: Union[str, Path],
    after_path: Union[str, Path],
    next_before_path: Union[str, Path],
    *,
    backend: str = "kimi",
    http_trust_env: bool = False,
    kimi_api_key: Optional[str] = None,
    kimi_base_url: str = DEFAULT_KIMI_BASE_URL,
    kimi_model: str = DEFAULT_KIMI_MODEL,
    kimi_max_tokens: int = 512,
    ark_api_key: Optional[str] = None,
    ark_base_url: str = DEFAULT_ARK_BASE_URL,
    ark_model: str = DEFAULT_ARK_MODEL,
    qwen_api_key: Optional[str] = None,
    qwen_base_url: str = DEFAULT_QWEN_BASE_URL,
    qwen_model: str = DEFAULT_QWEN_MODEL,
    gpt_api_key: Optional[str] = None,
    gpt_base_url: str = DEFAULT_GPT_BASE_URL,
    gpt_model: str = DEFAULT_GPT_MODEL,
) -> Dict[str, Any]:
    before = Path(before_path)
    after = Path(after_path)
    next_before = Path(next_before_path)
    if not before.is_file():
        raise FileNotFoundError(f"Before screenshot not found: {before}")
    if not after.is_file():
        raise FileNotFoundError(f"After screenshot not found: {after}")
    if not next_before.is_file():
        raise FileNotFoundError(f"Next action before screenshot not found: {next_before}")

    prompt_text = f"Action type: {action_type}"
    screenshots = [
        {"path": before, "label": "Screenshot before the action:"},
        {"path": after, "label": "Screenshot after the action:"},
        {"path": next_before, "label": "Screenshot before the next action:"},
    ]
    return judge_multimodal_json(
        system_prompt_text=system_prompt,
        prompt_text=prompt_text,
        screenshots=screenshots,
        backend=backend,
        http_trust_env=http_trust_env,
        max_tokens=kimi_max_tokens,
        compact_images=False,
        openai_use_system_message=False,
        provider_request_kwargs={
            "qwen": {"temperature": 0, "extra_body": {"enable_thinking": True}},
        },
        kimi_api_key=kimi_api_key,
        kimi_base_url=kimi_base_url,
        kimi_model=kimi_model,
        ark_api_key=ark_api_key,
        ark_base_url=ark_base_url,
        ark_model=ark_model,
        qwen_api_key=qwen_api_key,
        qwen_base_url=qwen_base_url,
        qwen_model=qwen_model,
        gpt_api_key=gpt_api_key,
        gpt_base_url=gpt_base_url,
        gpt_model=gpt_model,
    )
