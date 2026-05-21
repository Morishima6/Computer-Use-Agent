import os
from copy import deepcopy
from pathlib import Path

import httpx

import backoff
from anthropic import Anthropic
from openai import (
    AzureOpenAI,
    APIConnectionError,
    APIError,
    AzureOpenAI,
    OpenAI,
    RateLimitError,
)


_DEFAULT_HTTPX_CLIENT = None
_CURRENT_TASK_COST = None
_UNKNOWN_PRICING_MODELS = set()


def _normalize_model_name(model_name):
    if not model_name:
        return ""
    return str(model_name).strip().lower()


def _parse_price_per_1k(raw_value):
    if not raw_value:
        return None
    value = str(raw_value).strip()
    if "/" in value:
        value = value.split("/", 1)[0].strip()
    try:
        return float(value)
    except ValueError:
        return None


def _load_model_pricing():
    prices = {}
    cost_file = Path(__file__).resolve().parents[3] / "cost.md"
    if not cost_file.exists():
        return prices

    for line in cost_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue

        columns = [part.strip() for part in stripped.strip("|").split("|")]
        if len(columns) < 3:
            continue
        if columns[0].lower() in {"model", "-------"}:
            continue
        if set(columns[0]) == {"-"}:
            continue

        input_price = _parse_price_per_1k(columns[1])
        output_price = _parse_price_per_1k(columns[2])
        if input_price is None or output_price is None:
            continue

        prices[_normalize_model_name(columns[0])] = {
            "input_per_1k": input_price,
            "output_per_1k": output_price,
        }

    return prices


_MODEL_PRICING = _load_model_pricing()


def _resolve_pricing_model(model_name):
    normalized_model = _normalize_model_name(model_name)
    if not normalized_model:
        return None
    if normalized_model in _MODEL_PRICING:
        return normalized_model

    for priced_model in _MODEL_PRICING:
        if normalized_model.startswith(priced_model):
            return priced_model
    return None


def _extract_usage_value(usage, *names):
    if usage is None:
        return 0

    for name in names:
        if isinstance(usage, dict) and name in usage and usage[name] is not None:
            return usage[name]
        if hasattr(usage, name):
            value = getattr(usage, name)
            if value is not None:
                return value
    return 0


def reset_task_cost_tracking():
    global _CURRENT_TASK_COST
    _CURRENT_TASK_COST = {
        "total_cost": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "models": {},
    }


def get_task_cost_snapshot():
    if _CURRENT_TASK_COST is None:
        reset_task_cost_tracking()
    return deepcopy(_CURRENT_TASK_COST)


def _record_task_cost(model_name, usage):
    global _CURRENT_TASK_COST

    if _CURRENT_TASK_COST is None:
        reset_task_cost_tracking()

    prompt_tokens = int(
        _extract_usage_value(
            usage,
            "prompt_tokens",
            "input_tokens",
        )
        or 0
    )
    completion_tokens = int(
        _extract_usage_value(
            usage,
            "completion_tokens",
            "output_tokens",
        )
        or 0
    )
    total_tokens = int(
        _extract_usage_value(
            usage,
            "total_tokens",
        )
        or (prompt_tokens + completion_tokens)
    )

    resolved_model = _resolve_pricing_model(model_name)
    cost = 0.0
    if resolved_model:
        pricing = _MODEL_PRICING[resolved_model]
        cost = (
            (prompt_tokens / 1000.0) * pricing["input_per_1k"]
            + (completion_tokens / 1000.0) * pricing["output_per_1k"]
        )
    else:
        normalized_model = _normalize_model_name(model_name)
        if normalized_model and normalized_model not in _UNKNOWN_PRICING_MODELS:
            _UNKNOWN_PRICING_MODELS.add(normalized_model)

    model_key = resolved_model or _normalize_model_name(model_name) or "unknown"
    model_usage = _CURRENT_TASK_COST["models"].setdefault(
        model_key,
        {
            "model": model_name,
            "pricing_model": resolved_model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
        },
    )
    model_usage["prompt_tokens"] += prompt_tokens
    model_usage["completion_tokens"] += completion_tokens
    model_usage["total_tokens"] += total_tokens
    model_usage["cost"] += cost

    _CURRENT_TASK_COST["total_prompt_tokens"] += prompt_tokens
    _CURRENT_TASK_COST["total_completion_tokens"] += completion_tokens
    _CURRENT_TASK_COST["total_tokens"] += total_tokens
    _CURRENT_TASK_COST["total_cost"] += cost

    return cost


def _build_httpx_client() -> httpx.Client:
    """Create a shared httpx client honoring system proxy/CAs and avoiding TLS issues."""
    # Honor enterprise/corporate proxies and NO_PROXY from environment
    trust_env = True

    # Disable HTTP/2 to prevent TLS EOF issues with some proxies/MITM devices
    http2 = False

    # Allow custom CA bundle via common env vars, else default verification
    ca_bundle = (
        os.getenv("SSL_CERT_FILE")
        or os.getenv("REQUESTS_CA_BUNDLE")
        or os.getenv("CURL_CA_BUNDLE")
    )
    verify: bool | str = ca_bundle if ca_bundle else True

    # Conservative timeouts; separate connect vs read
    connect_timeout = float(os.getenv("HTTPX_CONNECT_TIMEOUT", "15"))
    read_timeout = float(os.getenv("HTTPX_READ_TIMEOUT", "60"))
    write_timeout = float(os.getenv("HTTPX_WRITE_TIMEOUT", "30"))
    pool_timeout = float(os.getenv("HTTPX_POOL_TIMEOUT", "60"))

    timeout = httpx.Timeout(
        connect=connect_timeout, read=read_timeout, write=write_timeout, pool=pool_timeout
    )

    # Enable built-in retries at the transport layer
    transport = httpx.HTTPTransport(retries=3)

    return httpx.Client(
        http2=http2,
        timeout=timeout,
        transport=transport,
        trust_env=trust_env,
        verify=verify,
    )


def _get_httpx_client() -> httpx.Client:
    global _DEFAULT_HTTPX_CLIENT
    if _DEFAULT_HTTPX_CLIENT is None:
        _DEFAULT_HTTPX_CLIENT = _build_httpx_client()
    return _DEFAULT_HTTPX_CLIENT


class LMMEngine:
    def __init__(self, model=None):
        self.model = model

    def _record_usage(self, usage):
        return _record_task_cost(self.model, usage)


class LMMEngineOpenAI(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        rate_limit=-1,
        temperature=None,
        organization=None,
        **kwargs,
    ):
        super().__init__(model=model)
        assert model is not None, "model must be provided"
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.organization = organization
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None
        self.temperature = temperature  # Can force temperature to be the same (in the case of o3 requiring temperature to be 1)

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named OPENAI_API_KEY"
            )
        organization = self.organization or os.getenv("OPENAI_ORG_ID")
        if not self.llm_client:
            if not self.base_url:
                self.llm_client = OpenAI(
                    api_key=api_key, organization=organization, http_client=_get_httpx_client()
                )
            else:
                self.llm_client = OpenAI(
                    base_url=self.base_url,
                    api_key=api_key,
                    organization=organization,
                    http_client=_get_httpx_client(),
                )
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            # max_completion_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=(
                temperature if self.temperature is None else self.temperature
            ),
            **kwargs,
        )
        self._record_usage(getattr(completion, "usage", None))
        return completion.choices[0].message.content


class LMMEngineAnthropic(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        thinking=False,
        temperature=None,
        **kwargs,
    ):
        super().__init__(model=model)
        assert model is not None, "model must be provided"
        self.model = model
        self.thinking = thinking
        self.api_key = api_key
        self.llm_client = None
        self.temperature = temperature

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named ANTHROPIC_API_KEY"
            )
        self.llm_client = Anthropic(api_key=api_key, http_client=_get_httpx_client())
        # Use the instance temperature if not specified in the call
        temp = self.temperature if temperature is None else temperature
        if self.thinking:
            full_response = self.llm_client.messages.create(
                system=messages[0]["content"][0]["text"],
                model=self.model,
                messages=messages[1:],
                max_tokens=8192,
                thinking={"type": "enabled", "budget_tokens": 4096},
                **kwargs,
            )
            self._record_usage(getattr(full_response, "usage", None))
            thoughts = full_response.content[0].thinking
            return full_response.content[1].text
        response = self.llm_client.messages.create(
            system=messages[0]["content"][0]["text"],
            model=self.model,
            messages=messages[1:],
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temp,
            **kwargs,
        )
        self._record_usage(getattr(response, "usage", None))
        return response.content[0].text

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    # Compatible with Claude-3.7 Sonnet thinking mode
    def generate_with_thinking(
        self, messages, temperature=0.0, max_new_tokens=None, **kwargs
    ):
        """Generate the next message based on previous messages, and keeps the thinking tokens"""
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named ANTHROPIC_API_KEY"
            )
        self.llm_client = Anthropic(api_key=api_key, http_client=_get_httpx_client())
        full_response = self.llm_client.messages.create(
            system=messages[0]["content"][0]["text"],
            model=self.model,
            messages=messages[1:],
            max_tokens=8192,
            thinking={"type": "enabled", "budget_tokens": 4096},
            **kwargs,
        )
        self._record_usage(getattr(full_response, "usage", None))

        thoughts = full_response.content[0].thinking
        answer = full_response.content[1].text
        full_response = (
            f"<thoughts>\n{thoughts}\n</thoughts>\n\n<answer>\n{answer}\n</answer>\n"
        )
        return full_response


class LMMEngineGemini(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        rate_limit=-1,
        temperature=None,
        **kwargs,
    ):
        super().__init__(model=model)
        assert model is not None, "model must be provided"
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None
        self.temperature = temperature

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("GEMINI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named GEMINI_API_KEY"
            )
        base_url = self.base_url or os.getenv("GEMINI_ENDPOINT_URL")
        if base_url is None:
            raise ValueError(
                "An endpoint URL needs to be provided in either the endpoint_url parameter or as an environment variable named GEMINI_ENDPOINT_URL"
            )
        if not self.llm_client:
            self.llm_client = OpenAI(
                base_url=base_url, api_key=api_key, http_client=_get_httpx_client()
            )
        # Use the temperature passed to generate, otherwise use the instance's temperature, otherwise default to 0.0
        temp = self.temperature if temperature is None else temperature
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temp,
            **kwargs,
        )
        self._record_usage(getattr(completion, "usage", None))
        return completion.choices[0].message.content


class LMMEngineOpenRouter(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        rate_limit=-1,
        temperature=None,
        **kwargs,
    ):
        super().__init__(model=model)
        assert model is not None, "model must be provided"
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None
        self.temperature = temperature

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named OPENROUTER_API_KEY"
            )
        base_url = self.base_url or os.getenv("OPEN_ROUTER_ENDPOINT_URL")
        if base_url is None:
            raise ValueError(
                "An endpoint URL needs to be provided in either the endpoint_url parameter or as an environment variable named OPEN_ROUTER_ENDPOINT_URL"
            )
        if not self.llm_client:
            self.llm_client = OpenAI(
                base_url=base_url, api_key=api_key, http_client=_get_httpx_client()
            )
        # Use self.temperature if set, otherwise use the temperature argument
        temp = self.temperature if self.temperature is not None else temperature
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temp,
            **kwargs,
        )
        self._record_usage(getattr(completion, "usage", None))
        return completion.choices[0].message.content


class LMMEngineAzureOpenAI(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        azure_endpoint=None,
        model=None,
        api_version=None,
        rate_limit=-1,
        temperature=None,
        **kwargs,
    ):
        super().__init__(model=model)
        assert model is not None, "model must be provided"
        self.model = model
        self.api_version = api_version
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None
        self.cost = 0.0
        self.temperature = temperature

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named AZURE_OPENAI_API_KEY"
            )
        api_version = self.api_version or os.getenv("OPENAI_API_VERSION")
        if api_version is None:
            raise ValueError(
                "api_version must be provided either as a parameter or as an environment variable named OPENAI_API_VERSION"
            )
        azure_endpoint = self.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if azure_endpoint is None:
            raise ValueError(
                "An Azure API endpoint needs to be provided in either the azure_endpoint parameter or as an environment variable named AZURE_OPENAI_ENDPOINT"
            )
        if not self.llm_client:
            self.llm_client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
                http_client=_get_httpx_client(),
            )
        # Use self.temperature if set, otherwise use the temperature argument
        temp = self.temperature if self.temperature is not None else temperature
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temp,
            **kwargs,
        )
        self.cost += self._record_usage(getattr(completion, "usage", None))
        return completion.choices[0].message.content


class LMMEnginevLLM(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        rate_limit=-1,
        temperature=None,
        **kwargs,
    ):
        super().__init__(model=model)
        assert model is not None, "model must be provided"
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None
        self.temperature = temperature

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(
        self,
        messages,
        temperature=0.0,
        top_p=0.8,
        repetition_penalty=1.05,
        max_new_tokens=512,
        **kwargs,
    ):
        api_key = self.api_key or os.getenv("vLLM_API_KEY")
        if api_key is None:
            raise ValueError(
                "A vLLM API key needs to be provided in either the api_key parameter or as an environment variable named vLLM_API_KEY"
            )
        base_url = self.base_url or os.getenv("vLLM_ENDPOINT_URL")
        if base_url is None:
            raise ValueError(
                "An endpoint URL needs to be provided in either the endpoint_url parameter or as an environment variable named vLLM_ENDPOINT_URL"
            )
        if not self.llm_client:
            self.llm_client = OpenAI(
                base_url=base_url, api_key=api_key, http_client=_get_httpx_client()
            )
        # Use self.temperature if set, otherwise use the temperature argument
        temp = self.temperature if self.temperature is not None else temperature
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temp,
            top_p=top_p,
            extra_body={"repetition_penalty": repetition_penalty},
        )
        self._record_usage(getattr(completion, "usage", None))
        return completion.choices[0].message.content


class LMMEngineHuggingFace(LMMEngine):
    def __init__(self, base_url=None, api_key=None, rate_limit=-1, **kwargs):
        super().__init__(model="tgi")
        self.base_url = base_url
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("HF_TOKEN")
        if api_key is None:
            raise ValueError(
                "A HuggingFace token needs to be provided in either the api_key parameter or as an environment variable named HF_TOKEN"
            )
        base_url = self.base_url or os.getenv("HF_ENDPOINT_URL")
        if base_url is None:
            raise ValueError(
                "HuggingFace endpoint must be provided as base_url parameter or as an environment variable named HF_ENDPOINT_URL."
            )
        if not self.llm_client:
            self.llm_client = OpenAI(base_url=base_url, api_key=api_key)
        completion = self.llm_client.chat.completions.create(
            model="tgi",
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature,
            **kwargs,
        )
        self._record_usage(getattr(completion, "usage", None))
        return completion.choices[0].message.content


class LMMEngineParasail(LMMEngine):
    def __init__(
        self, base_url=None, api_key=None, model=None, rate_limit=-1, **kwargs
    ):
        super().__init__(model=model)
        assert model is not None, "Parasail model id must be provided"
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("PARASAIL_API_KEY")
        if api_key is None:
            raise ValueError(
                "A Parasail API key needs to be provided in either the api_key parameter or as an environment variable named PARASAIL_API_KEY"
            )
        base_url = self.base_url
        if base_url is None:
            raise ValueError(
                "Parasail endpoint must be provided as base_url parameter or as an environment variable named PARASAIL_ENDPOINT_URL"
            )
        if not self.llm_client:
            self.llm_client = OpenAI(
                base_url=base_url if base_url else "https://api.parasail.io/v1",
                api_key=api_key,
                http_client=_get_httpx_client(),
            )
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature,
            **kwargs,
        )
        self._record_usage(getattr(completion, "usage", None))
        return completion.choices[0].message.content
