"""K128 test item rejected by the blind verifier as a near-duplicate, twice: dwell meter vs endcap gaze tracker (audit A8),
then speeding vehicle detector vs school-zone speed monitor. Neither may be an unsupported target while its twin is active."""
from src.edgebench.corpus.catalog import is_valid_unsupported_target
from src.edgebench.corpus.catalog_data import RAW_NOVEL_SERVICES, RAW_SERVICES


def _svc(sid):
    return next(s for s in RAW_SERVICES + RAW_NOVEL_SERVICES if s["id"] == sid)


def test_near_duplicate_targets_are_excluded_while_their_twin_is_active():
    for target, twin in [("customer_dwell_time_meter", "endcap_promotional_gaze_tracker"),
                         ("speeding_vehicle_detector", "school_zone_speed_monitor"),
                         ("h265_video_transcoder", "h264_bitrate_downscaler")]:
        assert is_valid_unsupported_target(_svc(target)) is True
        assert is_valid_unsupported_target(_svc(target), {twin, "count"}) is False
        assert is_valid_unsupported_target(_svc(target), {"count"}) is True
