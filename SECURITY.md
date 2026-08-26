# Security Policy

## Supported versions

GovAgent-KR is experimental alpha software. Security fixes are applied to the
latest `0.1.x` release and the `main` branch; older snapshots are unsupported.

## Reporting a vulnerability

Use this repository's **Security → Report a vulnerability** flow. Do not open a
public issue for credential exposure, sandbox escape, gold leakage, review-key
disclosure, path traversal, or a way to forge freeze/review evidence.

Include the affected commit or release, a minimal reproduction, expected impact,
and whether any secret or personal data may already have been exposed. Do not
attach real provider credentials, private reviewer identity maps, or sensitive
production transcripts. Maintainers will acknowledge a report through GitHub's
private advisory channel and coordinate disclosure after a fix is available.

## Security boundary

The SUT container is designed to have no network and no gold mount. Hosted model
traffic still leaves through the Inspect host and must be restricted by the
operator. This repository is an evaluation environment, not a production legal
service or a general-purpose secure sandbox.
