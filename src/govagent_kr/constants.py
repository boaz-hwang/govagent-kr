from pathlib import Path

PROFILE_ID = "govagent-kr-temporal-citation"
PROFILE_VERSION = "0.1.0"
SCHEMA_VERSION = "govagent-kr/v0.1"
CANONICALIZATION_VERSION = "quote-c14n/v1"
FIXTURE_PARSER_VERSION = "law-go-kr-xml/v1.1"

INSTRUMENT_ID = "KR-LAW-010098"
OFFICIAL_LAW_ID = "010098"
INSTRUMENT_NAME = "지방자치단체를 당사자로 하는 계약에 관한 법률 시행령"
TIMEZONE = "Asia/Seoul"

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_ROOT = PACKAGE_ROOT / "fixtures" / "local-contract-decree"
DEFAULT_CASES_PATH = PACKAGE_ROOT / "data" / "cases.jsonl"
DEFAULT_GOLD_PATH = PACKAGE_ROOT / "data" / "gold.jsonl"

SMOKE_ITEM_IDS = (
    "social-enterprise-direct-award--post_effective",
    "social-enterprise-single-quote--post_effective",
    "statutory-digital-service--post_effective",
    "legacy-innovation-transition--post_effective",
    "material-price-adjustment-transition--post_effective",
)

MAX_SUBMISSION_BYTES = 65_536
MAX_SELECTED_PROVISIONS = 32
MAX_MISSING_FACTS = 16
MAX_EXPLANATION_CHARS = 8_192
