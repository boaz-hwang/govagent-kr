from __future__ import annotations

from typing import Any

from .canonical import canonical_json_bytes, content_sha256

MODEL_CONFIG_SECTIONS = frozenset({"generate", "model_args"})
REQUIRED_GENERATE_CONFIG = {
    "max_connections": 1,
    "max_retries": 0,
    "timeout": 120,
    "attempt_timeout": 120,
}
FORBIDDEN_GENERATE_CAPABILITIES = frozenset(
    {"extra_body", "extra_headers", "fallback_models", "internal_tools"}
)
FORBIDDEN_MODEL_ARGUMENT_KEYS = frozenset(
    {
        "authorization",
        "base_url",
        "default_headers",
        "extra_headers",
        "headers",
        "model_base_url",
        "proxy_authorization",
        "x_api_key",
    }
)


def _normalized_key(value: Any) -> str:
    return str(value).casefold().replace("-", "_")


def _reject_model_argument_capabilities(value: Any, path: str = "model_args") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = _normalized_key(key)
            if normalized in FORBIDDEN_MODEL_ARGUMENT_KEYS:
                raise ValueError(
                    f"analysis model_config contains forbidden model argument: {path}.{key}"
                )
            _reject_model_argument_capabilities(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_model_argument_capabilities(child, f"{path}[{index}]")


def normalize_frozen_model_config(value: Any) -> dict[str, dict[str, Any]]:
    """Return the one canonical Inspect model configuration accepted for analysis."""

    try:
        from inspect_ai.model import GenerateConfig
    except ImportError as exc:  # pragma: no cover - analysis installs the Inspect extra
        raise ValueError("Inspect AI is required to validate an analysis model_config") from exc

    if not isinstance(value, dict) or not value:
        raise ValueError("analysis freeze requires a non-empty model_config object")
    if set(value) & MODEL_CONFIG_SECTIONS:
        unexpected = set(value) - MODEL_CONFIG_SECTIONS
        if unexpected:
            raise ValueError("sectioned model_config may contain only generate and model_args")
        generate_raw = value.get("generate", {})
        model_args = value.get("model_args", {})
        if not isinstance(generate_raw, dict) or not isinstance(model_args, dict):
            raise ValueError("model_config generate and model_args must be objects")
    else:
        generate_fields = set(GenerateConfig.model_fields)
        generate_raw = {key: child for key, child in value.items() if key in generate_fields}
        model_args = {key: child for key, child in value.items() if key not in generate_fields}

    _reject_model_argument_capabilities(model_args)
    if model_args:
        raise ValueError("analysis model_config model_args must be empty for the single-SUT MVP")
    try:
        generate = GenerateConfig.model_validate(generate_raw).model_dump(
            mode="json", exclude_none=True
        )
    except ValueError as exc:
        raise ValueError("analysis freeze has invalid Inspect generation settings") from exc
    forbidden = sorted(key for key in FORBIDDEN_GENERATE_CAPABILITIES if generate.get(key))
    if "extra_headers" in generate or forbidden:
        names = set(forbidden)
        if "extra_headers" in generate:
            names.add("extra_headers")
        raise ValueError(
            "analysis model_config contains a forbidden provider capability: "
            + ", ".join(sorted(names))
        )
    conflicts = {
        key: generate[key]
        for key, expected in REQUIRED_GENERATE_CONFIG.items()
        if key in generate and generate[key] != expected
    }
    if conflicts:
        raise ValueError(
            "model_config conflicts with the fixed Inspect controls: "
            + ", ".join(sorted(conflicts))
        )
    generate = {**REQUIRED_GENERATE_CONFIG, **generate}
    return {"generate": generate, "model_args": dict(model_args)}


def frozen_model_config_sha256(value: Any) -> str:
    return content_sha256(canonical_json_bytes(normalize_frozen_model_config(value)))


__all__ = [
    "REQUIRED_GENERATE_CONFIG",
    "frozen_model_config_sha256",
    "normalize_frozen_model_config",
]
