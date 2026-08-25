from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MAX_QUERY_CHARS = 256
MAX_FILE_BYTES = 16 * 1024 * 1024


class LawStoreError(RuntimeError):
    pass


class FixtureIntegrityError(LawStoreError):
    pass


class ToolInputError(LawStoreError):
    pass


def _read_verified_json(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    try:
        stat = path.stat()
        if stat.st_size > MAX_FILE_BYTES:
            raise FixtureIntegrityError(f"fixture file is too large: {path}")
        payload = path.read_bytes()
    except OSError as exc:
        raise FixtureIntegrityError(f"cannot read fixture file: {path}") from exc
    if expected_sha256 is not None:
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected_sha256:
            raise FixtureIntegrityError(f"fixture hash mismatch: {path}")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FixtureIntegrityError(f"invalid fixture JSON: {path}") from exc
    if not isinstance(value, dict):
        raise FixtureIntegrityError(f"fixture root must be an object: {path}")
    return value


def _public_provision(provision: dict[str, Any]) -> dict[str, str]:
    fields = ("instrument_id", "history_id", "provision_locator", "passage")
    try:
        result = {field: provision[field] for field in fields}
    except KeyError as exc:
        raise FixtureIntegrityError(f"provision is missing {exc.args[0]}") from exc
    if any(not isinstance(value, str) or not value for value in result.values()):
        raise FixtureIntegrityError("provision public fields must be non-empty strings")
    return result


class LawStore:
    """Read-only, hash-checked view of the public frozen-law fixture."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        index = _read_verified_json(self.root / "index.json")
        instrument_id = index.get("instrument_id")
        histories = index.get("histories")
        if not isinstance(instrument_id, str) or not instrument_id:
            raise FixtureIntegrityError("fixture index has no instrument_id")
        if not isinstance(histories, list) or not histories:
            raise FixtureIntegrityError("fixture index has no histories")
        self.instrument_id = instrument_id
        self.latest_history_id = index.get("latest_history_id")
        self._histories: dict[str, dict[str, Any]] = {}
        for history in histories:
            if not isinstance(history, dict):
                raise FixtureIntegrityError("fixture history must be an object")
            history_id = history.get("history_id")
            if not isinstance(history_id, str) or not history_id:
                raise FixtureIntegrityError("fixture history has no history_id")
            if history_id in self._histories:
                raise FixtureIntegrityError(f"duplicate history_id: {history_id}")
            self._snapshot_root(history)
            self._histories[history_id] = history
        if self.latest_history_id not in self._histories:
            raise FixtureIntegrityError("latest_history_id is not in histories")
        self._provisions: dict[str, dict[str, dict[str, Any]]] = {}

    def _snapshot_root(self, history: dict[str, Any]) -> Path:
        snapshot_dir = history.get("snapshot_dir")
        if not isinstance(snapshot_dir, str) or not snapshot_dir:
            raise FixtureIntegrityError("fixture history has no snapshot_dir")
        path = (self.root / snapshot_dir).resolve()
        if self.root not in path.parents:
            raise FixtureIntegrityError("fixture index attempts path traversal")
        return path

    def _load_history(self, history_id: str) -> dict[str, dict[str, Any]]:
        if history_id in self._provisions:
            return self._provisions[history_id]
        history = self._histories.get(history_id)
        if history is None:
            raise ToolInputError(f"unknown history_id: {history_id}")
        snapshot_root = self._snapshot_root(history)
        manifest_sha256 = history.get("manifest_sha256")
        if not isinstance(manifest_sha256, str):
            raise FixtureIntegrityError(f"history has no manifest hash: {history_id}")
        manifest = _read_verified_json(
            snapshot_root / "manifest.json", expected_sha256=manifest_sha256
        )
        artifact = manifest.get("artifacts", {}).get("normalized/provisions.json")
        if not isinstance(artifact, dict) or not isinstance(artifact.get("sha256"), str):
            raise FixtureIntegrityError(f"manifest lacks normalized artifact: {history_id}")
        normalized = _read_verified_json(
            snapshot_root / "normalized" / "provisions.json",
            expected_sha256=artifact["sha256"],
        )
        if normalized.get("history_id") != history_id:
            raise FixtureIntegrityError(f"normalized history_id mismatch: {history_id}")
        raw_provisions = normalized.get("provisions")
        if not isinstance(raw_provisions, list):
            raise FixtureIntegrityError(f"normalized provisions is not a list: {history_id}")
        provisions: dict[str, dict[str, Any]] = {}
        for provision in raw_provisions:
            if not isinstance(provision, dict):
                raise FixtureIntegrityError(f"invalid provision entry: {history_id}")
            locator = provision.get("provision_locator")
            if not isinstance(locator, str) or not locator:
                raise FixtureIntegrityError(f"provision has no locator: {history_id}")
            if locator in provisions:
                raise FixtureIntegrityError(f"duplicate provision locator: {history_id} {locator}")
            if provision.get("history_id") != history_id:
                raise FixtureIntegrityError(f"provision history mismatch: {history_id} {locator}")
            if provision.get("instrument_id") != self.instrument_id:
                raise FixtureIntegrityError(
                    f"provision instrument mismatch: {history_id} {locator}"
                )
            _public_provision(provision)
            provisions[locator] = provision
        self._provisions[history_id] = provisions
        return provisions

    def list_versions(self, instrument_id: str) -> list[dict[str, Any]]:
        if instrument_id != self.instrument_id:
            raise ToolInputError(f"unknown instrument_id: {instrument_id}")
        fields = (
            "history_id",
            "effective_date",
            "valid_to",
            "promulgation_number",
            "revision_type",
        )
        return [
            {field: history.get(field) for field in fields}
            for history in sorted(self._histories.values(), key=lambda item: item["effective_date"])
        ]

    def search_provisions(self, history_id: str, query: str) -> list[dict[str, str]]:
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARS:
            raise ToolInputError(f"query must be 1..{MAX_QUERY_CHARS} characters")
        terms = [term.casefold() for term in query.split() if term]
        if not terms:
            raise ToolInputError("query has no searchable terms")
        matches: list[tuple[int, str, dict[str, Any]]] = []
        for locator, provision in self._load_history(history_id).items():
            if provision.get("kind") in {"article", "supplement"}:
                continue
            haystack = f"{locator} {provision['passage']}".casefold()
            score = sum(haystack.count(term) for term in terms)
            if score:
                matches.append((-score, locator, provision))
        matches.sort(key=lambda item: (item[0], item[1]))
        return [_public_provision(item[2]) for item in matches[:20]]

    def get_provision(self, history_id: str, provision_locator: str) -> dict[str, str]:
        if not isinstance(provision_locator, str) or not provision_locator:
            raise ToolInputError("provision_locator must be a non-empty string")
        provision = self._load_history(history_id).get(provision_locator)
        if provision is None:
            raise ToolInputError(f"unknown provision: {history_id} {provision_locator}")
        return _public_provision(provision)

    def get_supplementary_provisions(self, history_id: str) -> list[dict[str, str]]:
        history = self._histories.get(history_id)
        if history is None:
            raise ToolInputError(f"unknown history_id: {history_id}")
        number = history.get("promulgation_number")
        if not isinstance(number, str) or not number:
            raise FixtureIntegrityError(f"history has no promulgation_number: {history_id}")
        prefix = f"부칙(대통령령 제{number}호)"
        return [
            _public_provision(provision)
            for locator, provision in sorted(self._load_history(history_id).items())
            if locator.startswith(prefix)
            and provision.get("kind") in {"supplement", "supplement_article"}
        ]


__all__ = [
    "FixtureIntegrityError",
    "LawStore",
    "LawStoreError",
    "ToolInputError",
]
