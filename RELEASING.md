# Release Process

## Alpha release

1. Confirm `git status --short` is empty.
2. Run the locked static, unit, replay, and Docker quality gates documented in
   `README.md`.
3. Confirm `govagent-report` says `engineering_environment_ready` and does not
   claim `analysis_ready` while gold is pending.
4. Review `DATASET_CARD.md`, `NOTICE`, and the secret scan for every added source.
5. Update `CHANGELOG.md`, package version, profile/data versions when applicable,
   and tag the immutable commit.

## Stable or validated benchmark release

An alpha tag must not be relabeled as stable merely because CI passes. Stable or
validated benchmark status additionally requires:

- two qualified independent legal reviews of every retained gold item;
- explicit adjudication of every disagreement and retained review evidence;
- a frozen five-item frontier smoke reviewed blind to grader output;
- the prespecified 20-item × 3-trial run and complete blind transcript review;
- a final report that passes the repository's fail-closed evidence checks;
- a new dataset/profile version if any legal text, case, gold, grader, or tool changes.

Never fabricate review identities, decisions, signatures, model runs, or receipts.
