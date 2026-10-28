"""
Unit test for ZoneIntrusionMonitor, DeepStreamMetadataAdapter, and schema compliance.
"""

import json
import os
import sys

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.zone_monitor import ZoneIntrusionMonitor, point_in_polygon
from src.deepstream_adapter import DeepStreamMetadataAdapter, DEEPSTREAM_UNTRACKED_ID


def test_point_in_polygon():
    polygon = [[0.2, 0.4], [0.6, 0.4], [0.6, 0.8], [0.2, 0.8]]
    # Center inside
    assert point_in_polygon((0.4, 0.6), polygon) is True
    # Clearly outside
    assert point_in_polygon((0.1, 0.1), polygon) is False
    assert point_in_polygon((0.8, 0.6), polygon) is False


def validate_schema(record: dict, schema: dict):
    # Verify required keys
    for req in schema.get("required", []):
        assert req in record, f"Missing required key in incident: {req}"

    # Verify types
    props = schema.get("properties", {})
    if "event_type" in record:
        assert record["event_type"] in props["event_type"]["enum"]
    if "severity" in record:
        assert record["severity"] in props["severity"]["enum"]
    if "confidence" in record:
        assert 0.0 <= record["confidence"] <= 1.0
    if "bbox" in record:
        assert len(record["bbox"]) == 4
        for val in record["bbox"]:
            assert 0.0 <= val <= 1.0
    if "ground_contact_point" in record:
        assert len(record["ground_contact_point"]) == 2
        for val in record["ground_contact_point"]:
            assert 0.0 <= val <= 1.0


def test_zone_monitor():
    test_log = "data/output/test_incident_records.jsonl"
    if os.path.exists(test_log):
        os.remove(test_log)

    with open("configs/incident_schema.json", "r", encoding="utf-8") as f:
        schema = json.load(f)

    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=test_log,
    )

    # 1. Detection outside restricted zone (polygon is x: 0.20-0.65, y: 0.45-0.90)
    incidents = monitor.process_detection(
        bbox=[0.05, 0.10, 0.15, 0.30],
        confidence=0.88,
        track_id=1,
        frame_index=1,
        timestamp_epoch=1000.0,
    )
    assert len(incidents) == 0, f"Expected 0 incidents outside, got {len(incidents)}"

    # 2. Person 1 enters Zone 1
    incidents = monitor.process_detection(
        bbox=[0.30, 0.40, 0.50, 0.70],
        confidence=0.92,
        track_id=1,
        frame_index=2,
        timestamp_epoch=1001.0,
    )
    assert len(incidents) == 1, f"Expected 1 intrusion incident, got {len(incidents)}"
    rec = incidents[0]
    validate_schema(rec, schema)
    assert rec["event_type"] == "ZONE_INTRUSION_ENTRY"
    assert rec["zone_id"] == "zone_restricted_01"
    assert rec["track_id"] == 1
    assert rec["confidence"] == 0.92

    # 3. Person 1 stays inside within cooldown (1.0s elapsed < 2.0s cooldown) -> no spam
    incidents_cooldown = monitor.process_detection(
        bbox=[0.31, 0.40, 0.51, 0.70],
        confidence=0.93,
        track_id=1,
        frame_index=3,
        timestamp_epoch=1002.0,
    )
    assert len(incidents_cooldown) == 0, "Cooldown failed to prevent alert spam"

    # 4. Person 1 stays inside after cooldown (3.0s elapsed > 2.0s cooldown) -> OCCUPANCY alert
    incidents_occ = monitor.process_detection(
        bbox=[0.31, 0.40, 0.51, 0.70],
        confidence=0.91,
        track_id=1,
        frame_index=4,
        timestamp_epoch=1004.5,
    )
    assert len(incidents_occ) == 1
    validate_schema(incidents_occ[0], schema)
    assert incidents_occ[0]["event_type"] == "ZONE_INTRUSION_OCCUPANCY"

    # 5. Person 1 moves outside -> EXIT event
    incidents_exit = monitor.process_detection(
        bbox=[0.05, 0.10, 0.15, 0.30],
        confidence=0.89,
        track_id=1,
        frame_index=5,
        timestamp_epoch=1005.0,
    )
    assert len(incidents_exit) == 1
    validate_schema(incidents_exit[0], schema)
    assert incidents_exit[0]["event_type"] == "ZONE_INTRUSION_EXIT"

    # Verify persisted JSON Lines log
    assert os.path.exists(test_log)
    with open(test_log, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 3
    for line in lines:
        validate_schema(line, schema)

    os.remove(test_log)


def test_deepstream_adapter():
    test_log = "data/output/test_adapter_records.jsonl"
    if os.path.exists(test_log):
        os.remove(test_log)

    with open("configs/incident_schema.json", "r", encoding="utf-8") as f:
        schema = json.load(f)

    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=test_log,
    )
    adapter = DeepStreamMetadataAdapter(
        zone_monitor=monitor,
        default_frame_width=1920,
        default_frame_height=1080,
    )

    # 1. Non-person detection (e.g. car class_id=2) -> should be ignored
    res = adapter.process_raw_dict({
        "class_id": 2,
        "confidence": 0.95,
        "rect_params": {"left": 500, "top": 500, "width": 300, "height": 300},
        "frame_width": 1920,
        "frame_height": 1080,
    })
    assert len(res) == 0

    # 2. Low confidence person (< 0.40) -> should be ignored
    res = adapter.process_raw_dict({
        "class_id": 0,
        "confidence": 0.25,
        "rect_params": {"left": 500, "top": 500, "width": 300, "height": 300},
    })
    assert len(res) == 0

    # 3. Person in pixel space inside zone (left=600, top=500, width=200, height=300)
    # in 1920x1080 -> x: [600/1920=0.3125, 800/1920=0.4167], y: [500/1080=0.463, 800/1080=0.741]
    # Ground point: ((0.3125+0.4167)/2, 0.741) = (0.3646, 0.741) -> INSIDE Zone 1 (x: 0.20-0.65, y: 0.45-0.90)
    res = adapter.process_raw_dict({
        "class_id": 0,
        "confidence": 0.87,
        "rect_params": {"left": 600, "top": 500, "width": 200, "height": 300},
        "frame_width": 1920,
        "frame_height": 1080,
        "track_id": DEEPSTREAM_UNTRACKED_ID,  # Sentinel untracked
        "frame_index": 10,
        "timestamp": 2000.0,
    })
    assert len(res) == 1
    rec = res[0]
    validate_schema(rec, schema)
    assert rec["track_id"] is None  # Verified cleaned sentinel
    assert rec["event_type"] == "ZONE_INTRUSION_ENTRY"
    assert rec["zone_id"] == "zone_restricted_01"

    if os.path.exists(test_log):
        os.remove(test_log)

    print("ALL TESTS PASSED: Geometric containment, state transitions, cooldown anti-spam, DeepStream pixel normalization, non-person filtering, and schema validation.")


if __name__ == "__main__":
    test_point_in_polygon()
    test_zone_monitor()
    test_deepstream_adapter()
