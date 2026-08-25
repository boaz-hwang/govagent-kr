from govagent_kr.fixtures import SNAPSHOTS, fixture_integrity_report


def test_frozen_fixture_is_anchored_to_raw_hashes_and_parser_replay() -> None:
    report = fixture_integrity_report()

    assert report["ok"] is True
    assert report["history_count"] == len(SNAPSHOTS) == 7
    assert report["checked_artifacts"] == 14
    assert report["pinned_raw_hashes_checked"] == 7
    assert report["normalized_replays_checked"] == 7
    assert report["errors"] == []
