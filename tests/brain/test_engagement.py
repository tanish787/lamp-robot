from brain.engagement import EngagementDebouncer


def test_no_transition_before_hold_time_elapses():
    d = EngagementDebouncer(hold_seconds=0.75)
    assert d.update(True, now=0.0) is None
    assert d.update(True, now=0.5) is None


def test_engages_once_hold_time_elapses():
    d = EngagementDebouncer(hold_seconds=0.75)
    d.update(True, now=0.0)
    assert d.update(True, now=0.8) is True


def test_does_not_re_fire_while_already_engaged():
    d = EngagementDebouncer(hold_seconds=0.75)
    d.update(True, now=0.0)
    d.update(True, now=0.8)
    assert d.update(True, now=1.5) is None


def test_brief_glance_away_does_not_disengage():
    d = EngagementDebouncer(hold_seconds=0.75)
    d.update(True, now=0.0)
    d.update(True, now=0.8)  # now engaged
    d.update(False, now=0.9)  # a 0.2s glance away
    assert d.update(True, now=1.0) is None  # still engaged, no flicker


def test_disengages_once_absence_hold_time_elapses():
    d = EngagementDebouncer(hold_seconds=0.75)
    d.update(True, now=0.0)
    d.update(True, now=0.8)  # engaged
    d.update(False, now=0.9)
    assert d.update(False, now=1.7) is False  # 0.8s absent, past the hold


def test_does_not_re_fire_while_already_disengaged():
    d = EngagementDebouncer(hold_seconds=0.75)
    d.update(False, now=0.0)
    assert d.update(False, now=10.0) is None
