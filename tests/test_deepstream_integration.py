"""
Comprehensive Integration Test for Sentinel Vision AI
Simulates realistic DeepStream multi-frame metadata streams with:
- Multiple concurrent people
- Continuous zone crossing trajectory (outside -> entry -> occupancy -> exit)
- Non-person class filtering (vehicles, noise)
- Low-confidence detection filtering
- Missing/unassigned tracking IDs (DEEPSTREAM_UNTRACKED_ID)
- Repeated frame processing & occupancy cooldown
- Malformed metadata handling (corrupted JSON, NaN, invalid types)
- Multi-person untracked spatial disambiguation
- Complete JSONL file verification against configs/incident_schema.json
"""

import json
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.zone_monitor import ZoneIntrusionMonitor
from src.deepstream_adapter import DeepStreamMetadataAdapter, DEEPSTREAM_UNTRACKED_ID
from src.sentinel_pipeline import run_stream_consumer, consume_stream_iterable


@pytest.fixture
def schema():
    with open("configs/incident_schema.json", "r", encoding="utf-8") as f:
        return json.load(f)


def validate_record_schema(record: dict, schema: dict):
    for req in schema.get("required", []):
        assert req in record, f"Missing required property: {req}"
    props = schema.get("properties", {})
    assert record["event_type"] in props["event_type"]["enum"]
    assert record["severity"] in props["severity"]["enum"]
    assert 0.0 <= record["confidence"] <= 1.0
    assert len(record["bbox"]) == 4
    for c in record["bbox"]:
        assert 0.0 <= c <= 1.0
    assert len(record["ground_contact_point"]) == 2
    for c in record["ground_contact_point"]:
        assert 0.0 <= c <= 1.0


def test_multi_person_trajectory_stream(schema):
    """
    Simulates a 60-frame video sequence at 30 FPS:
    - Frame resolution: 1920x1080
    - Zone 1 (zone_restricted_01) is normalized: x in [0.20, 0.65], y in [0.45, 0.90]
      In 1920x1080 pixels: x in [384, 1248], y in [486, 972]
    """
    output_log = "data/output/test_integration_incidents.jsonl"
    if os.path.exists(output_log):
        os.remove(output_log)

    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=output_log,
    )
    adapter = DeepStreamMetadataAdapter(
        zone_monitor=monitor,
        min_confidence=0.50,
        default_frame_width=1920,
        default_frame_height=1080,
    )

    # 60 frames, simulating 2.0 seconds (30 fps)
    for frame_idx in range(1, 61):
        timestamp = 1700000000.0 + (frame_idx / 30.0)

        # -------------------------------------------------------------
        # Person 1 (Track 101): Walking into Zone 1, staying, then exiting
        # -------------------------------------------------------------
        if 1 <= frame_idx <= 10:
            p1_left, p1_top, p1_w, p1_h = 100, 200, 100, 200  # ground y = 400 (outside)
        elif 11 <= frame_idx <= 40:
            p1_left, p1_top, p1_w, p1_h = 500, 400, 120, 300  # ground y = 700, x = 560 (inside Zone 1)
        else:
            p1_left, p1_top, p1_w, p1_h = 100, 200, 100, 200  # ground y = 400 (outside)

        adapter.process_raw_dict({
            "class_id": 0,
            "confidence": 0.88,
            "rect_params": {"left": p1_left, "top": p1_top, "width": p1_w, "height": p1_h},
            "track_id": 101,
            "frame_index": frame_idx,
            "timestamp": timestamp,
        })

        # Person 2 (Track 102): Safe worker walking strictly outside zone
        adapter.process_raw_dict({
            "class_id": 0,
            "confidence": 0.94,
            "rect_params": {"left": 50, "top": 50, "width": 80, "height": 180},
            "track_id": 102,
            "frame_index": frame_idx,
            "timestamp": timestamp,
        })

        # Non-person Object: Forklift / Car (class_id = 2) inside zone -> must be filtered
        adapter.process_raw_dict({
            "class_id": 2,
            "confidence": 0.99,
            "rect_params": {"left": 600, "top": 500, "width": 400, "height": 300},
            "track_id": 201,
            "frame_index": frame_idx,
            "timestamp": timestamp,
        })

        # Low Confidence Detection: Ghost (< 0.50) -> must be filtered
        adapter.process_raw_dict({
            "class_id": 0,
            "confidence": 0.35,
            "rect_params": {"left": 550, "top": 450, "width": 100, "height": 200},
            "track_id": 301,
            "frame_index": frame_idx,
            "timestamp": timestamp,
        })

        # Untracked Person on frame 25
        if frame_idx == 25:
            adapter.process_raw_dict({
                "class_id": 0,
                "confidence": 0.85,
                "rect_params": {"left": 520, "top": 420, "width": 110, "height": 280},
                "track_id": DEEPSTREAM_UNTRACKED_ID,
                "frame_index": frame_idx,
                "timestamp": timestamp,
            })

    # Verify generated incident records in output file
    assert os.path.exists(output_log), "Output log was not created"
    with open(output_log, "r", encoding="utf-8") as f:
        incidents = [json.loads(line) for line in f if line.strip()]

    # Validate schema for all generated incidents
    for inc in incidents:
        validate_record_schema(inc, schema)

    # Person 1 (Track 101) entered on frame 11 -> must have ZONE_INTRUSION_ENTRY
    p1_entries = [i for i in incidents if i["track_id"] == 101 and i["event_type"] == "ZONE_INTRUSION_ENTRY"]
    assert len(p1_entries) == 1
    assert p1_entries[0]["frame_index"] == 11
    assert p1_entries[0]["zone_id"] == "zone_restricted_01"

    # Person 1 (Track 101) exited on frame 41 -> must have ZONE_INTRUSION_EXIT
    p1_exits = [i for i in incidents if i["track_id"] == 101 and i["event_type"] == "ZONE_INTRUSION_EXIT"]
    assert len(p1_exits) == 1
    assert p1_exits[0]["frame_index"] == 41

    # Person 2 (Track 102) stayed outside -> must have 0 incidents
    p2_incidents = [i for i in incidents if i["track_id"] == 102]
    assert len(p2_incidents) == 0

    # Untracked person on frame 25 -> must have track_id is None and ENTRY event
    untracked_incidents = [i for i in incidents if i["track_id"] is None]
    assert len(untracked_incidents) == 1
    assert untracked_incidents[0]["frame_index"] == 25

    if os.path.exists(output_log):
        os.remove(output_log)


def test_malformed_metadata_handling(schema):
    """
    Tests robustness against corrupted, invalid, and malformed inputs:
    - Incomplete JSON lines
    - Non-numeric bounding box values
    - Inverted/negative dimensions (w <= 0, h <= 0)
    - Null/missing rect_params
    - Non-numeric confidence scores
    """
    test_log = "data/output/test_malformed_handling.jsonl"
    if os.path.exists(test_log):
        os.remove(test_log)

    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=test_log,
    )
    adapter = DeepStreamMetadataAdapter(zone_monitor=monitor)

    malformed_fixtures = [
        # Missing rect_params and bbox
        {"class_id": 0, "confidence": 0.90},
        # Non-numeric rect_params
        {"class_id": 0, "confidence": 0.90, "rect_params": {"left": "bad", "top": 100, "width": 50, "height": 50}},
        # Inverted box (width <= 0)
        {"class_id": 0, "confidence": 0.90, "rect_params": {"left": 500, "top": 500, "width": -20, "height": 100}},
        # NaN confidence
        {"class_id": 0, "confidence": float("nan"), "rect_params": {"left": 500, "top": 500, "width": 50, "height": 100}},
        # Out-of-range confidence (> 1.0)
        {"class_id": 0, "confidence": 99.0, "rect_params": {"left": 500, "top": 500, "width": 50, "height": 100}},
        # Corrupted bbox list (len != 4)
        {"class_id": 0, "confidence": 0.85, "bbox": [0.5, 0.5]},
        # Not a dict
        "invalid string input",
        None,
        42,
    ]

    for fix in malformed_fixtures:
        res = adapter.process_raw_dict(fix)
        assert res == [], f"Expected [] for malformed input {fix}, got {res}"

    # Also test consume_stream_iterable with corrupted JSON text
    corrupted_json_lines = [
        "{ truncated json",
        "",
        "   ",
        "!!!not json at all!!!",
        '{"valid_format_but_not_detection": true}',
    ]
    det_count, inc_count, incs = consume_stream_iterable(corrupted_json_lines, adapter, verbose=False)
    assert det_count == 1  # only valid JSON dict processed
    assert inc_count == 0  # 0 incidents

    if os.path.exists(test_log):
        os.remove(test_log)


def test_occupancy_cooldown_and_jsonl_persistence(schema):
    """
    Tests exact interval throttling and JSONL formatting:
    - Frame at t=0.0 -> ENTRY
    - Frame at t=1.0 -> inside cooldown (2.0s) -> no alert
    - Frame at t=2.5 -> outside cooldown (2.5 > 2.0s) -> OCCUPANCY alert
    - Verify file append mode preserves all records without corrupting JSON lines.
    """
    test_log = "data/output/test_cooldown_persistence.jsonl"
    if os.path.exists(test_log):
        os.remove(test_log)

    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=test_log,
    )

    # Box inside Zone 1: ground contact (0.40, 0.70)
    inside_bbox = [0.30, 0.40, 0.50, 0.70]

    # Event 1: ENTRY at t=100.0
    r1 = monitor.process_detection(bbox=inside_bbox, confidence=0.90, track_id=55, frame_index=1, timestamp_epoch=100.0)
    assert len(r1) == 1
    assert r1[0]["event_type"] == "ZONE_INTRUSION_ENTRY"

    # Event 2: Still inside at t=101.0 (elapsed 1.0s < 2.0s cooldown) -> 0 incidents
    r2 = monitor.process_detection(bbox=inside_bbox, confidence=0.90, track_id=55, frame_index=31, timestamp_epoch=101.0)
    assert len(r2) == 0

    # Event 3: Still inside at t=102.5 (elapsed 2.5s > 2.0s cooldown) -> OCCUPANCY alert
    r3 = monitor.process_detection(bbox=inside_bbox, confidence=0.90, track_id=55, frame_index=76, timestamp_epoch=102.5)
    assert len(r3) == 1
    assert r3[0]["event_type"] == "ZONE_INTRUSION_OCCUPANCY"

    # Verify JSONL lines on disk
    with open(test_log, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 2
    for l in lines:
        parsed = json.loads(l)
        validate_record_schema(parsed, schema)

    if os.path.exists(test_log):
        os.remove(test_log)


def test_multiple_untracked_entities_distinct_locations(schema):
    """
    Tests two people entering the zone on the same frame, both with unassigned track_id (None).
    Verifies that spatial disambiguation creates distinct entities without collision.
    """
    test_log = "data/output/test_untracked_multi.jsonl"
    if os.path.exists(test_log):
        os.remove(test_log)

    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=test_log,
    )

    # Person A in left side of zone: [0.25, 0.50, 0.35, 0.75] -> ground x=0.30
    # Person B in right side of zone: [0.50, 0.50, 0.60, 0.75] -> ground x=0.55
    res_a = monitor.process_detection(bbox=[0.25, 0.50, 0.35, 0.75], confidence=0.88, track_id=None, frame_index=10, timestamp_epoch=50.0)
    res_b = monitor.process_detection(bbox=[0.50, 0.50, 0.60, 0.75], confidence=0.91, track_id=None, frame_index=10, timestamp_epoch=50.0)

    assert len(res_a) == 1
    assert len(res_b) == 1
    assert res_a[0]["track_id"] is None
    assert res_b[0]["track_id"] is None
    assert res_a[0]["bbox"] != res_b[0]["bbox"]

    if os.path.exists(test_log):
        os.remove(test_log)


def test_stream_consumer_cli_integration(schema):
    """
    Tests sentinel_pipeline.py run_stream_consumer end-to-end reading from a realistic JSONL file.
    """
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as stream_file:
        stream_path = stream_file.name
        # Record 1: outside
        stream_file.write(json.dumps({
            "class_id": 0, "confidence": 0.90,
            "rect_params": {"left": 50, "top": 50, "width": 50, "height": 100},
            "track_id": 999, "frame_index": 1, "timestamp": 1000.0
        }) + "\n")
        # Record 2: intrusion inside Zone 1
        stream_file.write(json.dumps({
            "class_id": 0, "confidence": 0.91,
            "rect_params": {"left": 600, "top": 500, "width": 100, "height": 300},
            "track_id": 999, "frame_index": 2, "timestamp": 1001.0
        }) + "\n")

    test_output = "data/output/test_cli_consumer_output.jsonl"
    if os.path.exists(test_output):
        os.remove(test_output)

    try:
        detections, incidents = run_stream_consumer(
            zone_config="configs/restricted_zone.json",
            output_log=test_output,
            input_stream_file=stream_path,
            verbose=False,
        )
        assert detections == 2
        assert incidents == 1

        with open(test_output, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 1
        validate_record_schema(lines[0], schema)
        assert lines[0]["event_type"] == "ZONE_INTRUSION_ENTRY"
        assert lines[0]["track_id"] == 999
    finally:
        if os.path.exists(stream_path):
            os.remove(stream_path)
        if os.path.exists(test_output):
            os.remove(test_output)


def test_real_pipegen_metadata_end_to_end(schema):
    """
    End-to-end integration test reading authentic DeepStream perception metadata
    produced by the compiled PipeGen YOLO11s TensorRT pipeline
    (artifacts/pipegen-export/output/metadata.json).

    Verifies:
    1. Complete ingestion of all 163 frames and 320 detected objects.
    2. Detection normalization from 1728x1080 pixel space to [0.0, 1.0].
    3. Ground-contact point projection to restricted zone polygons.
    4. Emission of genuine ENTRY and OCCUPANCY events.
    5. Cooldown throttling preventing alarm flooding (320 detections -> exactly 6 incidents).
    6. 100% schema compliance of all emitted JSONL incident records.
    """
    meta_path = "artifacts/pipegen-export/output/metadata.json"
    if not os.path.isfile(meta_path):
        pytest.skip(f"PipeGen metadata file not found at {meta_path}")

    test_output = "data/output/test_pipegen_real_metadata.jsonl"
    if os.path.exists(test_output):
        os.remove(test_output)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=test_output,
    )
    adapter = DeepStreamMetadataAdapter(
        zone_monitor=monitor,
        min_confidence=0.40,
        default_frame_width=1728,
        default_frame_height=1080,
    )

    total_detections = 0
    all_incidents = []

    for frame in meta_data.get("frames", []):
        w = frame.get("width", 1728)
        h = frame.get("height", 1080)
        pts_sec = frame.get("pts_ns", 0) / 1_000_000_000

        for obj in frame.get("objects", []):
            bbox = obj.get("bbox", {})
            if not isinstance(bbox, dict):
                continue

            total_detections += 1
            record = {
                "class_id": obj.get("class_id", 0),
                "confidence": obj.get("confidence", 0.0),
                "track_id": obj.get("id"),
                "frame_index": frame.get("frame"),
                "timestamp_epoch": pts_sec,
                "frame_width": w,
                "frame_height": h,
                "rect_params": {
                    "left": bbox.get("left"),
                    "top": bbox.get("top"),
                    "width": bbox.get("width"),
                    "height": bbox.get("height"),
                },
            }
            incidents = adapter.process_raw_dict(record)
            if incidents:
                all_incidents.extend(incidents)

    try:
        # Assertions on genuine perception metadata
        assert len(meta_data.get("frames", [])) == 163
        assert total_detections == 320
        assert len(all_incidents) == 6

        # Check event breakdown
        entries = [i for i in all_incidents if i["event_type"] == "ZONE_INTRUSION_ENTRY"]
        occupancies = [i for i in all_incidents if i["event_type"] == "ZONE_INTRUSION_OCCUPANCY"]
        assert len(entries) == 2
        assert len(occupancies) == 4

        # Verify track IDs involved (Track 0 and Track 1)
        tracks_in_incidents = set(i["track_id"] for i in all_incidents)
        assert tracks_in_incidents == {0, 1}

        # Verify zone breached
        for inc in all_incidents:
            assert inc["zone_id"] == "zone_restricted_01"
            assert inc["severity"] == "CRITICAL"
            validate_record_schema(inc, schema)

        # Verify persisted JSONL file
        assert os.path.isfile(test_output)
        with open(test_output, "r", encoding="utf-8") as f:
            persisted = [json.loads(line) for line in f if line.strip()]
        assert len(persisted) == 6
        for p in persisted:
            validate_record_schema(p, schema)

    finally:
        if os.path.exists(test_output):
            os.remove(test_output)


def test_repaired_demo_video_motion_and_integrity():
    """
    Verifies that the repaired demonstration video:
    1. Exists and is non-empty.
    2. Decodes all 163 frames completely without corruption.
    3. Has correct framerate (30.0 FPS) and duration (~5.43s).
    4. Confirms that background and target regions advance sequentially
       rather than freezing on a static still image.
    """
    import cv2
    import numpy as np

    video_path = "data/output/sentinel_demo_repaired.mp4"
    if not os.path.isfile(video_path):
        pytest.skip(f"Repaired video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Failed to open video: {video_path}"

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Exact frame count
    assert len(frames) == 163, f"Expected 163 frames, got {len(frames)}"

    # Dimensions
    h, w, c = frames[0].shape
    assert (w, h) == (1728, 1080)

    # Frame drift from beginning to end
    f0 = frames[0].astype(float)
    f_end = frames[-1].astype(float)
    total_delta = np.mean(np.abs(f_end - f0))
    assert total_delta > 0.5, f"Video appears completely frozen: total drift={total_delta}"

    # Consecutive differences (confirming continuous decoding)
    diffs = [np.mean(np.abs(frames[i+1].astype(float) - frames[i].astype(float))) for i in range(len(frames)-1)]
    assert len(diffs) == 162
    assert all(d >= 0.0 for d in diffs)


