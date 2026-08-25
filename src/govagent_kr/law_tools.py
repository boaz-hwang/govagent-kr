from __future__ import annotations

from typing import Any

from .fixtures import FixtureStore


def _public_provision(provision: dict[str, Any]) -> dict[str, Any]:
    return {
        "instrument_id": provision["instrument_id"],
        "history_id": provision["history_id"],
        "provision_locator": provision["provision_locator"],
        "passage": provision["passage"],
    }


class LawTools:
    """The four deterministic, offline tool contracts exposed to the SUT."""

    def __init__(self, store: FixtureStore) -> None:
        self.store = store

    def list_versions(self, instrument_id: str) -> list[dict[str, Any]]:
        versions = self.store.list_versions(instrument_id)
        return [
            {
                "history_id": item["history_id"],
                "effective_date": item["effective_date"],
                "valid_to": item["valid_to"],
                "promulgation_number": item["promulgation_number"],
                "revision_type": item["revision_type"],
            }
            for item in versions
        ]

    def search_provisions(self, history_id: str, query: str) -> list[dict[str, Any]]:
        return [
            _public_provision(item)
            for item in self.store.search_provisions(history_id, query, limit=20)
        ]

    def get_provision(self, history_id: str, provision_locator: str) -> dict[str, Any]:
        return _public_provision(self.store.get_provision(history_id, provision_locator))

    def get_supplementary_provisions(self, history_id: str) -> list[dict[str, Any]]:
        return [
            _public_provision(item) for item in self.store.get_supplementary_provisions(history_id)
        ]


TOOL_NAME_MAPPING = {
    "law.list_versions": "law_list_versions",
    "law.search_provisions": "law_search_provisions",
    "law.get_provision": "law_get_provision",
    "law.get_supplementary_provisions": "law_get_supplementary_provisions",
}


__all__ = ["TOOL_NAME_MAPPING", "LawTools"]
