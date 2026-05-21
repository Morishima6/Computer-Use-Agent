from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from openai import OpenAI


PROVIDER_CONFIGS: dict[str, dict[str, Any]] = {
    "openai_compatible": {
        "default_base_url": "https://api.chatanywhere.tech/v1",
        "default_model": "kimi-k2.5",
        "api_env_names": ("KIMI_API_KEY", "MOONSHOT_API_KEY", "OPENAI_API_KEY"),
        "base_url_env_names": ("KIMI_BASE_URL", "MOONSHOT_BASE_URL", "OPENAI_BASE_URL"),
    },
    "kimi_coding": {
        "default_base_url": "https://api.kimi.com/coding/",
        "default_model": "kimi-2.5",
        "api_env_names": ("KIMI_CODING_API_KEY", "KIMI_API_KEY"),
        "base_url_env_names": ("KIMI_CODING_BASE_URL",),
    },
    "ark_coding": {
        "default_base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
        "default_model": "Kimi-K2.5",
        "api_env_names": ("ARK_CODING_API_KEY", "ARK_API_KEY"),
        "base_url_env_names": ("ARK_CODING_BASE_URL", "ARK_BASE_URL"),
    },
}

SUPPORTED_PROVIDERS = tuple(PROVIDER_CONFIGS.keys())
PROVIDER_ALIASES = {
    "openai": "openai_compatible",
    "compat": "openai_compatible",
    "kimi": "kimi_coding",
    "ark": "ark_coding",
    "volc": "ark_coding",
}


def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def resolve_provider_api_key(provider: str, explicit_api_key: str) -> str:
    provider = normalize_provider_name(provider)
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


def resolve_provider_base_url(provider: str, explicit_base_url: str) -> str:
    provider = normalize_provider_name(provider)
    if explicit_base_url:
        return explicit_base_url
    config = PROVIDER_CONFIGS[provider]
    for env_name in config["base_url_env_names"]:
        value = os.getenv(env_name)
        if value:
            return value
    return str(config["default_base_url"])


def resolve_provider_model(provider: str, explicit_model: str) -> str:
    provider = normalize_provider_name(provider)
    if explicit_model:
        return explicit_model
    return str(PROVIDER_CONFIGS[provider]["default_model"])


def normalize_provider_name(provider: str) -> str:
    normalized = provider.strip().lower()
    return PROVIDER_ALIASES.get(normalized, normalized)


def _normalize_openai_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return "\n".join(part for part in text_parts if part)
    return str(content)


def _build_openai_multimodal_user_content(
    prompt: str,
    before_path: Path,
    after_path: Path,
) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": prompt},
        {"type": "text", "text": "Image 1 is the BEFORE screenshot."},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image(before_path)}"},
        },
        {"type": "text", "text": "Image 2 is the AFTER screenshot."},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image(after_path)}"},
        },
    ]


def _call_openai_compatible(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    prompt: str,
    before_path: Path,
    after_path: Path,
    max_output_tokens: int,
) -> dict[str, Any]:
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": _build_openai_multimodal_user_content(
                    prompt=prompt,
                    before_path=before_path,
                    after_path=after_path,
                ),
            },
        ],
        temperature=0.0,
        max_tokens=max_output_tokens,
    )
    message = response.choices[0].message
    return {
        "raw_text": _normalize_openai_content(message.content),
        "reasoning": getattr(message, "reasoning_content", None),
        "provider_meta": {
            "provider": "openai_compatible",
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", model),
        },
    }


def _call_ark_coding(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    prompt: str,
    before_path: Path,
    after_path: Path,
    max_output_tokens: int,
    enable_thinking: bool,
) -> dict[str, Any]:
    client = OpenAI(api_key=api_key, base_url=base_url)
    extra_body: dict[str, Any] | None = None
    if enable_thinking:
        extra_body = {"thinking": {"type": "enabled"}}
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": _build_openai_multimodal_user_content(
                    prompt=prompt,
                    before_path=before_path,
                    after_path=after_path,
                ),
            },
        ],
        max_tokens=max_output_tokens,
        extra_body=extra_body,
    )
    message = response.choices[0].message
    return {
        "raw_text": _normalize_openai_content(message.content),
        "reasoning": getattr(message, "reasoning_content", None),
        "provider_meta": {
            "provider": "ark_coding",
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", model),
        },
    }


def _call_kimi_coding(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    prompt: str,
    before_path: Path,
    after_path: Path,
    max_output_tokens: int,
    enable_thinking: bool,
    thinking_budget_tokens: int,
) -> dict[str, Any]:
    normalized_base = base_url.rstrip("/")
    if normalized_base.endswith("/messages"):
        url = normalized_base
    elif normalized_base.endswith("/v1"):
        url = normalized_base + "/messages"
    else:
        url = normalized_base + "/v1/messages"
    payload: dict[str, Any] = {
        "model": model,
        "system": system_prompt,
        "max_tokens": max_output_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "Image 1 is the BEFORE screenshot."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": encode_image(before_path),
                        },
                    },
                    {"type": "text", "text": "Image 2 is the AFTER screenshot."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": encode_image(after_path),
                        },
                    },
                ],
            }
        ],
    }
    if enable_thinking:
        payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget_tokens}

    request = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"kimi_coding_request_failed: {exc.code} {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"kimi_coding_request_failed: {exc}") from exc

    text_blocks: list[str] = []
    thinking_blocks: list[str] = []
    for block in data.get("content", []):
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text_blocks.append(str(block.get("text", "")))
        elif block_type == "thinking":
            thinking_blocks.append(str(block.get("thinking", "")))

    return {
        "raw_text": "\n".join(part for part in text_blocks if part).strip(),
        "reasoning": "\n\n".join(part for part in thinking_blocks if part).strip() or None,
        "provider_meta": {
            "provider": "kimi_coding",
            "response_id": data.get("id"),
            "model": data.get("model", model),
            "stop_reason": data.get("stop_reason"),
        },
    }


def call_audit_model(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    prompt: str,
    before_path: Path,
    after_path: Path,
    max_output_tokens: int = 1024,
    enable_thinking: bool = True,
    thinking_budget_tokens: int = 10000,
) -> dict[str, Any]:
    provider = normalize_provider_name(provider)
    if provider == "openai_compatible":
        return _call_openai_compatible(
            api_key=api_key,
            base_url=base_url,
            model=model,
            system_prompt=system_prompt,
            prompt=prompt,
            before_path=before_path,
            after_path=after_path,
            max_output_tokens=max_output_tokens,
        )
    if provider == "ark_coding":
        return _call_ark_coding(
            api_key=api_key,
            base_url=base_url,
            model=model,
            system_prompt=system_prompt,
            prompt=prompt,
            before_path=before_path,
            after_path=after_path,
            max_output_tokens=max_output_tokens,
            enable_thinking=enable_thinking,
        )
    if provider == "kimi_coding":
        return _call_kimi_coding(
            api_key=api_key,
            base_url=base_url,
            model=model,
            system_prompt=system_prompt,
            prompt=prompt,
            before_path=before_path,
            after_path=after_path,
            max_output_tokens=max_output_tokens,
            enable_thinking=enable_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
        )
    raise ValueError(f"Unsupported provider: {provider}")
