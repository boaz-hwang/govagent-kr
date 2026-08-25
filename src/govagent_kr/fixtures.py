from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from lxml import etree

from .canonical import canonical_json_bytes, content_sha256, quote_sha256
from .constants import (
    DEFAULT_FIXTURE_ROOT,
    FIXTURE_PARSER_VERSION,
    INSTRUMENT_ID,
    INSTRUMENT_NAME,
    OFFICIAL_LAW_ID,
    PROFILE_VERSION,
)

OFFICIAL_API_ENDPOINT = "https://www.law.go.kr/DRF/lawService.do"
OFFICIAL_PAGE_ENDPOINT = "https://www.law.go.kr/LSW/lsInfoP.do"


@dataclass(frozen=True)
class SnapshotSpec:
    lsi_seq: str
    effective_date: str
    valid_to: str | None
    promulgation_date: str
    promulgation_number: str
    revision_type: str
    boundary_id: str | None
    role: str
    expected_law_key: str
    expected_raw_sha256: str | None

    @property
    def history_id(self) -> str:
        return f"{INSTRUMENT_ID}:{self.lsi_seq}:{self.effective_date}"

    @property
    def snapshot_dirname(self) -> str:
        return f"{self.lsi_seq}-{self.effective_date}"

    @property
    def official_page_url(self) -> str:
        query = urlencode({"lsiSeq": self.lsi_seq, "efYd": self.effective_date})
        return f"{OFFICIAL_PAGE_ENDPOINT}?{query}"


SNAPSHOTS: tuple[SnapshotSpec, ...] = (
    SnapshotSpec(
        lsi_seq="201260",
        effective_date="20180118",
        valid_to="20180724",
        promulgation_date="20180116",
        promulgation_number="28586",
        revision_type="타법개정",
        boundary_id="social-enterprise-2018-07-24",
        role="pre",
        expected_law_key="0100982018011628586",
        expected_raw_sha256="a4d0077321c3bb73950c3a77c33c64a753dec75b0719cbb686f219c96e7ead3f",
    ),
    SnapshotSpec(
        lsi_seq="204108",
        effective_date="20180724",
        valid_to="20181213",
        promulgation_date="20180724",
        promulgation_number="29059",
        revision_type="일부개정",
        boundary_id="social-enterprise-2018-07-24",
        role="post",
        expected_law_key="0100982018072429059",
        expected_raw_sha256="44e0d1099402aa7ee4c2a89e5c8ad2920eea1dc359137a575cf451b5c7f389db",
    ),
    SnapshotSpec(
        lsi_seq="221951",
        effective_date="20201001",
        valid_to="20201103",
        promulgation_date="20200929",
        promulgation_number="31053",
        revision_type="타법개정",
        boundary_id="digital-service-2020-11-03",
        role="pre",
        expected_law_key="0100982020092931053",
        expected_raw_sha256="acfa0a13900d94265f8ec343b863203ec04be14c246993c08c6485ef8885c1c6",
    ),
    SnapshotSpec(
        lsi_seq="222683",
        effective_date="20201103",
        valid_to="20201210",
        promulgation_date="20201103",
        promulgation_number="31136",
        revision_type="일부개정",
        boundary_id="digital-service-2020-11-03",
        role="post",
        expected_law_key="0100982020110331136",
        expected_raw_sha256="ad34913749d0dcca700d2442773a841e7323472f15ec8cdb1326bda60acc3c93",
    ),
    SnapshotSpec(
        lsi_seq="269761",
        effective_date="20250312",
        valid_to="20250708",
        promulgation_date="20250312",
        promulgation_number="35382",
        revision_type="타법개정",
        boundary_id="price-adjustment-2025-07-08",
        role="pre",
        expected_law_key="0100982025031235382",
        expected_raw_sha256="566557082e472ce4b86b3c94dc7addb45191438c03fc0a0b2331dff232abcfb4",
    ),
    SnapshotSpec(
        lsi_seq="272723",
        effective_date="20250708",
        valid_to="20251001",
        promulgation_date="20250708",
        promulgation_number="35644",
        revision_type="일부개정",
        boundary_id="price-adjustment-2025-07-08",
        role="post",
        expected_law_key="0100982025070835644",
        expected_raw_sha256="ca0aa2d0f3467ba93b9ec3736e8012986ea309576b30d121026877690412569d",
    ),
    SnapshotSpec(
        lsi_seq="286149",
        effective_date="20260603",
        valid_to=None,
        promulgation_date="20260519",
        promulgation_number="36338",
        revision_type="타법개정",
        boundary_id=None,
        role="latest",
        expected_law_key="0100982026051936338",
        expected_raw_sha256="c01875d131ed0964e8ac26542c11261f561716657917c349073d5d934be261df",
    ),
)


class FixtureBuildError(RuntimeError):
    pass


class FixtureLookupError(KeyError):
    pass


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _xml_request(spec: SnapshotSpec, oc: str) -> tuple[bytes, dict[str, str]]:
    parameters = {
        "OC": oc,
        "target": "eflaw",
        "MST": spec.lsi_seq,
        "efYd": spec.effective_date,
        "type": "XML",
    }
    request = Request(
        f"{OFFICIAL_API_ENDPOINT}?{urlencode(parameters)}",
        headers={"User-Agent": "GovAgent-KR-Eval/0.1 fixture-freezer"},
    )
    with urlopen(request, timeout=60) as response:
        raw = response.read()
        media_type = response.headers.get_content_type()
    if media_type not in {"application/xml", "text/xml", "text/plain"}:
        raise FixtureBuildError(f"unexpected official API media type: {media_type}")
    public_parameters = dict(parameters)
    public_parameters["OC"] = "test" if oc == "test" else "<redacted-issued-key>"
    return raw, public_parameters


def _child_text(node: etree._Element, name: str) -> str:
    child = node.find(name)
    if child is None:
        return ""
    return _normalize_source_text("\n".join(child.itertext()))


def _normalize_source_text(value: str) -> str:
    lines: list[str] = []
    for raw_line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"[\t\f\v ]+", " ", raw_line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


_CIRCLED_NUMBERS = {character: index for index, character in enumerate("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳", 1)}


def _article_locator(article: etree._Element) -> str:
    number = _child_text(article, "조문번호")
    branch = _child_text(article, "조문가지번호")
    return f"제{number}조" + (f"의{branch}" if branch and branch != "0" else "")


def _paragraph_suffix(raw: str) -> str:
    value = raw.strip()
    if value in _CIRCLED_NUMBERS:
        return f"제{_CIRCLED_NUMBERS[value]}항"
    match = re.fullmatch(r"(\d+)", value)
    if match:
        return f"제{match.group(1)}항"
    raise FixtureBuildError(f"unsupported paragraph number: {raw!r}")


def _item_suffix(raw: str) -> str:
    value = raw.strip().rstrip(".")
    match = re.fullmatch(r"(\d+)(?:의(\d+))?", value)
    if not match:
        raise FixtureBuildError(f"unsupported item number: {raw!r}")
    return f"제{match.group(1)}호" + (f"의{match.group(2)}" if match.group(2) else "")


def _subitem_suffix(raw: str) -> str:
    value = raw.strip().rstrip(".")
    if not re.fullmatch(r"[가-힣]+", value):
        raise FixtureBuildError(f"unsupported subitem number: {raw!r}")
    return f"{value}목"


def _provision(
    *,
    history_id: str,
    locator: str,
    passage: str,
    kind: str,
    source_key: str,
) -> dict[str, Any]:
    if not passage:
        raise FixtureBuildError(f"empty passage at {history_id} {locator}")
    return {
        "instrument_id": INSTRUMENT_ID,
        "history_id": history_id,
        "provision_locator": locator,
        "passage": passage,
        "exact_passage_hash": quote_sha256(passage),
        "kind": kind,
        "source_key": source_key,
    }


def _embedded_enumeration_provisions(
    *, history_id: str, base_locator: str, passage: str, source_key: str
) -> list[dict[str, Any]]:
    """Split plaintext ``1)`` lists that the official XML leaves inside 목내용."""

    matches = list(re.finditer(r"(?m)^(\d+(?:의\d+)?)\)\s+", passage))
    if not matches:
        return []
    markers = [match.group(1) for match in matches]
    if len(markers) != len(set(markers)):
        raise FixtureBuildError(f"duplicate embedded marker at {history_id} {base_locator}")
    result: list[dict[str, Any]] = []
    prefix = passage[: matches[0].start()].rstrip("\n")
    if prefix:
        result.append(
            _provision(
                history_id=history_id,
                locator=f"{base_locator}#본문",
                passage=prefix,
                kind="embedded_preamble",
                source_key=source_key,
            )
        )
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(passage)
        item_passage = passage[match.start() : end].rstrip("\n")
        result.append(
            _provision(
                history_id=history_id,
                locator=f"{base_locator}#{match.group(1)})",
                passage=item_passage,
                kind="embedded_item",
                source_key=source_key,
            )
        )
    return result


def _parse_regular_provisions(root: etree._Element, spec: SnapshotSpec) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for article in root.findall("./조문/조문단위"):
        if _child_text(article, "조문여부") != "조문":
            continue
        locator = _article_locator(article)
        source_key = article.get("조문키", "")
        article_parts = [_child_text(article, "조문내용")]
        for paragraph in article.findall("./항"):
            paragraph_text = _child_text(paragraph, "항내용")
            paragraph_number = _child_text(paragraph, "항번호")
            paragraph_locator = (
                locator + _paragraph_suffix(paragraph_number) if paragraph_number else locator
            )
            article_parts.append(paragraph_text)
            if paragraph_number:
                result.append(
                    _provision(
                        history_id=spec.history_id,
                        locator=paragraph_locator,
                        passage=paragraph_text,
                        kind="paragraph",
                        source_key=source_key,
                    )
                )
            for item in paragraph.findall("./호"):
                item_text = _child_text(item, "호내용")
                item_locator = paragraph_locator + _item_suffix(_child_text(item, "호번호"))
                article_parts.append(item_text)
                result.append(
                    _provision(
                        history_id=spec.history_id,
                        locator=item_locator,
                        passage=item_text,
                        kind="item",
                        source_key=source_key,
                    )
                )
                for subitem in item.findall("./목"):
                    subitem_text = _child_text(subitem, "목내용")
                    subitem_locator = item_locator + _subitem_suffix(_child_text(subitem, "목번호"))
                    article_parts.append(subitem_text)
                    result.append(
                        _provision(
                            history_id=spec.history_id,
                            locator=subitem_locator,
                            passage=subitem_text,
                            kind="subitem",
                            source_key=source_key,
                        )
                    )
                    result.extend(
                        _embedded_enumeration_provisions(
                            history_id=spec.history_id,
                            base_locator=subitem_locator,
                            passage=subitem_text,
                            source_key=source_key,
                        )
                    )
        for item in article.findall("./호"):
            item_text = _child_text(item, "호내용")
            item_locator = locator + _item_suffix(_child_text(item, "호번호"))
            article_parts.append(item_text)
            result.append(
                _provision(
                    history_id=spec.history_id,
                    locator=item_locator,
                    passage=item_text,
                    kind="item",
                    source_key=source_key,
                )
            )
            for subitem in item.findall("./목"):
                subitem_text = _child_text(subitem, "목내용")
                subitem_locator = item_locator + _subitem_suffix(_child_text(subitem, "목번호"))
                article_parts.append(subitem_text)
                result.append(
                    _provision(
                        history_id=spec.history_id,
                        locator=subitem_locator,
                        passage=subitem_text,
                        kind="subitem",
                        source_key=source_key,
                    )
                )
                result.extend(
                    _embedded_enumeration_provisions(
                        history_id=spec.history_id,
                        base_locator=subitem_locator,
                        passage=subitem_text,
                        source_key=source_key,
                    )
                )
        article_passage = "\n".join(part for part in article_parts if part)
        result.append(
            _provision(
                history_id=spec.history_id,
                locator=locator,
                passage=article_passage,
                kind="article",
                source_key=source_key,
            )
        )
    return result


def _parse_supplementary_provisions(
    root: etree._Element, spec: SnapshotSpec
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for supplement in root.findall("./부칙/부칙단위"):
        number = _child_text(supplement, "부칙공포번호")
        content_node = supplement.find("부칙내용")
        if not number or content_node is None:
            continue
        # The task exposes only the supplementary provisions promulgated with
        # this frozen history. Older supplements remain in the raw XML but are
        # not duplicated into the normalized runtime corpus.
        if number != spec.promulgation_number:
            continue
        source_key = supplement.get("부칙키", "")
        lines: list[str] = []
        for text_node in content_node.itertext():
            normalized = _normalize_source_text(text_node)
            if normalized:
                lines.extend(normalized.split("\n"))
        if not lines:
            continue
        base_locator = f"부칙(대통령령 제{number}호)"
        result.append(
            _provision(
                history_id=spec.history_id,
                locator=base_locator,
                passage="\n".join(lines),
                kind="supplement",
                source_key=source_key,
            )
        )
        for line in lines:
            match = re.match(r"^제(\d+)조(?:\([^)]*\))?\s+", line)
            if not match:
                continue
            result.append(
                _provision(
                    history_id=spec.history_id,
                    locator=f"{base_locator}제{match.group(1)}조",
                    passage=line,
                    kind="supplement_article",
                    source_key=source_key,
                )
            )
    return result


def parse_official_xml(
    raw: bytes, spec: SnapshotSpec
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False, huge_tree=True)
    try:
        root = etree.fromstring(raw, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise FixtureBuildError(f"invalid official XML for {spec.history_id}") from exc
    if root.tag != "법령":
        raise FixtureBuildError(f"official API returned unexpected root {root.tag!r}")
    metadata = {
        "official_law_id": _child_text(root, "./기본정보/법령ID"),
        "law_key": root.get("법령키", ""),
        "name": _child_text(root, "./기본정보/법령명_한글"),
        "effective_date": _child_text(root, "./기본정보/시행일자"),
        "promulgation_date": _child_text(root, "./기본정보/공포일자"),
        "promulgation_number": _child_text(root, "./기본정보/공포번호"),
        "revision_type": _child_text(root, "./기본정보/제개정구분"),
    }
    expected = {
        "official_law_id": OFFICIAL_LAW_ID,
        "law_key": spec.expected_law_key,
        "name": INSTRUMENT_NAME,
        "effective_date": spec.effective_date,
        "promulgation_date": spec.promulgation_date,
        "promulgation_number": spec.promulgation_number,
        "revision_type": spec.revision_type,
    }
    if metadata != expected:
        raise FixtureBuildError(
            f"official metadata mismatch for {spec.history_id}: {metadata!r} != {expected!r}"
        )
    provisions = _parse_regular_provisions(root, spec)
    provisions.extend(_parse_supplementary_provisions(root, spec))
    identities = [record["provision_locator"] for record in provisions]
    if len(identities) != len(set(identities)):
        duplicates = sorted({item for item in identities if identities.count(item) > 1})
        raise FixtureBuildError(f"duplicate normalized locators: {duplicates[:10]}")
    provisions.sort(key=lambda item: item["provision_locator"])
    return metadata, provisions


def build_fixtures(
    output_root: Path = DEFAULT_FIXTURE_ROOT,
    *,
    oc: str = "test",
    force: bool = False,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FixtureBuildError(f"fixture directory is not empty: {output_root}; pass force=True")
    output_root.mkdir(parents=True, exist_ok=True)
    histories: list[dict[str, Any]] = []
    for spec in SNAPSHOTS:
        raw, public_parameters = _xml_request(spec, oc)
        raw_hash = content_sha256(raw)
        if spec.expected_raw_sha256 and raw_hash != spec.expected_raw_sha256:
            raise FixtureBuildError(
                f"official raw hash changed for {spec.history_id}: "
                f"expected {spec.expected_raw_sha256}, got {raw_hash}"
            )
        metadata, provisions = parse_official_xml(raw, spec)
        snapshot_root = output_root / spec.snapshot_dirname
        raw_path = snapshot_root / "raw" / "official.xml"
        normalized_path = snapshot_root / "normalized" / "provisions.json"
        _atomic_write(raw_path, raw)
        normalized_payload = canonical_json_bytes(
            {
                "fixture_parser_version": FIXTURE_PARSER_VERSION,
                "history_id": spec.history_id,
                "provisions": provisions,
            }
        )
        _atomic_write(normalized_path, normalized_payload)
        manifest = {
            "profile_version": PROFILE_VERSION,
            "fixture_parser_version": FIXTURE_PARSER_VERSION,
            "instrument_id": INSTRUMENT_ID,
            "history_id": spec.history_id,
            "official_metadata": metadata,
            "valid_from": date.fromisoformat(spec.effective_date).isoformat(),
            "valid_to": date.fromisoformat(spec.valid_to).isoformat() if spec.valid_to else None,
            "boundary_id": spec.boundary_id,
            "role": spec.role,
            "official_page_url": spec.official_page_url,
            "download": {
                "endpoint": OFFICIAL_API_ENDPOINT,
                "method": "GET",
                "parameters": public_parameters,
                "retrieved_on": "2026-08-25",
            },
            "artifacts": {
                "raw/official.xml": {
                    "sha256": raw_hash,
                    "bytes": len(raw),
                    "media_type": "application/xml",
                    "encoding": "utf-8",
                },
                "normalized/provisions.json": {
                    "sha256": content_sha256(normalized_payload),
                    "bytes": len(normalized_payload),
                    "media_type": "application/json",
                    "encoding": "utf-8",
                },
            },
            "provision_count": len(provisions),
            "license": {
                "basis": "대한민국 저작권법 제7조 공공저작물 비보호 대상 법령",
                "source_attribution": "법제처 국가법령정보센터",
            },
        }
        manifest_payload = canonical_json_bytes(manifest)
        _atomic_write(snapshot_root / "manifest.json", manifest_payload)
        histories.append(
            {
                "history_id": spec.history_id,
                "effective_date": date.fromisoformat(spec.effective_date).isoformat(),
                "valid_to": (
                    date.fromisoformat(spec.valid_to).isoformat() if spec.valid_to else None
                ),
                "promulgation_number": spec.promulgation_number,
                "revision_type": spec.revision_type,
                "boundary_id": spec.boundary_id,
                "role": spec.role,
                "snapshot_dir": spec.snapshot_dirname,
                "manifest_sha256": content_sha256(manifest_payload),
            }
        )
    index = {
        "profile_version": PROFILE_VERSION,
        "instrument_id": INSTRUMENT_ID,
        "instrument_name": INSTRUMENT_NAME,
        "latest_history_id": next(spec.history_id for spec in SNAPSHOTS if spec.role == "latest"),
        "histories": histories,
    }
    index_payload = canonical_json_bytes(index)
    _atomic_write(output_root / "index.json", index_payload)
    return {**index, "index_sha256": content_sha256(index_payload)}


class FixtureStore:
    def __init__(self, root: Path = DEFAULT_FIXTURE_ROOT) -> None:
        self.root = root.resolve()
        try:
            self.index = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise FixtureLookupError(f"invalid fixture index at {self.root}") from exc
        if self.index.get("instrument_id") != INSTRUMENT_ID:
            raise FixtureLookupError("fixture instrument ID mismatch")
        self._history: dict[str, dict[str, Any]] = {
            item["history_id"]: item for item in self.index["histories"]
        }
        self._provisions: dict[str, dict[str, dict[str, Any]]] = {}

    @property
    def latest_history_id(self) -> str:
        return str(self.index["latest_history_id"])

    def _load_history(self, history_id: str) -> dict[str, dict[str, Any]]:
        if history_id in self._provisions:
            return self._provisions[history_id]
        history = self._history.get(history_id)
        if history is None:
            raise FixtureLookupError(f"unknown history_id: {history_id}")
        snapshot_dir = (self.root / history["snapshot_dir"]).resolve()
        if self.root not in snapshot_dir.parents:
            raise FixtureLookupError("fixture index attempts path traversal")
        path = snapshot_dir / "normalized" / "provisions.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise FixtureLookupError(f"invalid normalized fixture: {path}") from exc
        if payload.get("history_id") != history_id:
            raise FixtureLookupError(f"normalized history ID mismatch: {path}")
        provisions = {item["provision_locator"]: item for item in payload["provisions"]}
        if len(provisions) != len(payload["provisions"]):
            raise FixtureLookupError(f"duplicate locators in {path}")
        self._provisions[history_id] = provisions
        return provisions

    def list_versions(self, instrument_id: str) -> list[dict[str, Any]]:
        if instrument_id != INSTRUMENT_ID:
            raise FixtureLookupError(f"unknown instrument_id: {instrument_id}")
        return [
            dict(item) for item in sorted(self._history.values(), key=lambda x: x["effective_date"])
        ]

    def get_provision(self, history_id: str, locator: str) -> dict[str, Any]:
        try:
            return dict(self._load_history(history_id)[locator])
        except KeyError as exc:
            raise FixtureLookupError(f"unknown provision: {history_id} {locator}") from exc

    def maybe_get_provision(self, history_id: str, locator: str) -> dict[str, Any] | None:
        try:
            return self.get_provision(history_id, locator)
        except FixtureLookupError:
            return None

    def search_provisions(
        self, history_id: str, query: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        if not query or limit < 1 or limit > 100:
            raise FixtureLookupError("query must be non-empty and limit must be 1..100")
        terms = [term.casefold() for term in query.split() if term]
        if not terms:
            raise FixtureLookupError("query has no searchable terms")
        matches: list[tuple[int, str, dict[str, Any]]] = []
        for locator, provision in self._load_history(history_id).items():
            if provision["kind"] in {"article", "supplement"}:
                continue
            haystack = f"{locator} {provision['passage']}".casefold()
            score = sum(haystack.count(term) for term in terms)
            if score:
                matches.append((-score, locator, dict(provision)))
        matches.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in matches[:limit]]

    def get_supplementary_provisions(self, history_id: str) -> list[dict[str, Any]]:
        history = self._history.get(history_id)
        if history is None:
            raise FixtureLookupError(f"unknown history_id: {history_id}")
        number = history["promulgation_number"]
        prefix = f"부칙(대통령령 제{number}호)"
        return [
            dict(provision)
            for locator, provision in sorted(self._load_history(history_id).items())
            if locator.startswith(prefix)
            and provision["kind"] in {"supplement", "supplement_article"}
        ]

    def iter_provisions(self) -> Iterable[dict[str, Any]]:
        for history_id in sorted(self._history):
            yield from self._load_history(history_id).values()


def fixture_integrity_report(root: Path = DEFAULT_FIXTURE_ROOT) -> dict[str, Any]:
    store = FixtureStore(root)
    errors: list[str] = []
    checked_artifacts = 0
    pinned_raw_hashes_checked = 0
    normalized_replays_checked = 0
    specs_by_directory = {spec.snapshot_dirname: spec for spec in SNAPSHOTS}
    indexed_directories = {history["snapshot_dir"] for history in store.index["histories"]}
    if indexed_directories != set(specs_by_directory):
        errors.append("fixture index snapshot set differs from the pinned specification")
    for history in store.index["histories"]:
        snapshot_root = root / history["snapshot_dir"]
        spec = specs_by_directory.get(history["snapshot_dir"])
        if spec is None:
            errors.append(f"fixture index contains an unpinned snapshot: {snapshot_root}")
            continue
        manifest_path = snapshot_root / "manifest.json"
        try:
            manifest_payload = manifest_path.read_bytes()
            if content_sha256(manifest_payload) != history["manifest_sha256"]:
                errors.append(f"manifest hash mismatch: {manifest_path}")
            manifest = json.loads(manifest_payload)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            errors.append(f"invalid manifest {manifest_path}: {exc}")
            continue
        for relative, expected in manifest["artifacts"].items():
            artifact_path = snapshot_root / relative
            try:
                payload = artifact_path.read_bytes()
            except FileNotFoundError:
                errors.append(f"missing artifact: {artifact_path}")
                continue
            checked_artifacts += 1
            if len(payload) != expected["bytes"]:
                errors.append(f"byte count mismatch: {artifact_path}")
            if content_sha256(payload) != expected["sha256"]:
                errors.append(f"sha256 mismatch: {artifact_path}")
        raw_path = snapshot_root / "raw" / "official.xml"
        normalized_path = snapshot_root / "normalized" / "provisions.json"
        try:
            raw_payload = raw_path.read_bytes()
            normalized_payload = normalized_path.read_bytes()
        except FileNotFoundError:
            continue
        if spec.expected_raw_sha256 is None:
            errors.append(f"official raw hash is not pinned in code: {raw_path}")
        elif content_sha256(raw_payload) != spec.expected_raw_sha256:
            errors.append(f"official raw differs from the code-pinned hash: {raw_path}")
        else:
            pinned_raw_hashes_checked += 1
        try:
            _metadata, replayed_provisions = parse_official_xml(raw_payload, spec)
        except FixtureBuildError as exc:
            errors.append(f"official raw replay failed for {raw_path}: {exc}")
            continue
        expected_normalized = canonical_json_bytes(
            {
                "fixture_parser_version": FIXTURE_PARSER_VERSION,
                "history_id": spec.history_id,
                "provisions": replayed_provisions,
            }
        )
        if normalized_payload != expected_normalized:
            errors.append(f"normalized fixture differs from parser replay: {normalized_path}")
        else:
            normalized_replays_checked += 1
    return {
        "ok": not errors,
        "history_count": len(store.index["histories"]),
        "checked_artifacts": checked_artifacts,
        "pinned_raw_hashes_checked": pinned_raw_hashes_checked,
        "normalized_replays_checked": normalized_replays_checked,
        "errors": errors,
    }


__all__ = [
    "SNAPSHOTS",
    "FixtureBuildError",
    "FixtureLookupError",
    "FixtureStore",
    "SnapshotSpec",
    "build_fixtures",
    "fixture_integrity_report",
    "parse_official_xml",
]
