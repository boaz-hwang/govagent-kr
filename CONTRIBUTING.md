# Contributing to GovAgent-KR

GovAgent-KR is a deliberately narrow evaluation task pack. Contributions are
welcome when they preserve its evidence-first and fail-closed design.

## Before opening a pull request

1. Open or reference a GitHub issue describing the observable problem.
2. Keep the change within the temporal-citation evaluation scope.
3. Do not include personal data, real administrative records, provider keys,
   reviewer identity maps, model logs, or non-redistributable source material.
4. For legal-text or gold corrections, cite the official instrument, history ID,
   effective date, provision, and source URL. Candidate gold is not promoted to
   adjudicated gold without two independent qualified reviews and a recorded
   adjudication.
5. Anyone refreshing official API fixtures must use their own approved access,
   keep credentials outside Git, and preserve source attribution and hashes.

## Development setup

```bash
uv sync --frozen --extra dev --extra inspect --python 3.12
uv run govagent-validate
uv run govagent-diagnostics
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

Docker and Compose are required for the isolated Agent Bridge tests:

```bash
GOVAGENT_RUN_DOCKER_SMOKE=1 uv run pytest -q tests/test_inspect_integration.py
```

Pull requests must include tests for changed behavior. A grader change should
include an adversarial regression that fails before the fix. Do not weaken a
fail-closed validation merely to make a fixture or transcript pass.

## Contribution license

By submitting a contribution, you agree that it may be distributed under the
Apache License 2.0 and that you have the right to submit it. Official legal text
and third-party material retain the status described in `NOTICE`.
