"""
Sentinel Vision AI - Software Simulation & Logic Verification Demo

DISCLAIMER:
This script is a software simulation and integration test fixture designed to
verify Sentinel Vision AI's spatial zone-intrusion logic, anti-spam cooldown,
and structured JSON incident generation.
This is NOT real GPU inference. Real DeepStream/TensorRT inference is executed
via `launch_demo.py` once the PipeGen container finishes initialization.
"""

import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.zone_monitor import ZoneIntrusionMonitor
from src.deepstream_adapter import DeepStreamMetadataAdapter, DEEPSTREAM_UNTRACKED_ID


def run_simulation_demo():
    print("\n" + "=" * 80)
    print("      SENTINEL VISION AI — SOFTWARE SIMULATION & VERIFICATION DEMO")
    print("=" * 80)
    print("DISCLAIMER: This script demonstrates spatial zone containment, state")
    print("transitions, and JSON Lines alerting logic using realistic simulated")
    print("test fixtures. This is NOT real GPU inference.")
    print("=" * 80 + "\n")

    zone_config = os.path.join(PROJECT_ROOT, "configs", "restricted_zone.json")
    output_log = os.path.join(PROJECT_ROOT, "data", "output", "simulation_incident_records.jsonl")
    schema_file = os.path.join(PROJECT_ROOT, "configs", "incident_schema.json")

    # Clean previous run
    if os.path.exists(output_log):
        os.remove(output_log)

    with open(schema_file, "r", encoding="utf-8") as f:
        schema = json.load(f)

    monitor = ZoneIntrusionMonitor(config_path=zone_config, output_log_path=output_log)
    adapter = DeepStreamMetadataAdapter(
        zone_monitor=monitor,
        default_frame_width=1920,
        default_frame_height=1080,
    )

    print(f"[INIT] Loaded Zone Configuration: {zone_config}")
    print(f"[INIT] Emitting Incident Records to: {output_log}\n")

    # Simulation Scenario:
    # 30 frames representing 1.0 second of video at 30 FPS
    # Worker A (Track 101): Walks into Zone 1 on frame 10, stays inside, exits on frame 25
    # Worker B (Track 102): Safe worker, stays outside
    # Worker C (Untracked): Enters on frame 15 without a track ID
    total_incidents = []

    print("[SIMULATION] Running 30-frame simulated scenario...")
    for frame_idx in range(1, 31):
        ts = 1700000000.0 + (frame_idx / 30.0)

        # Worker A (Track 101)
        if frame_idx < 10:
            wa_left, wa_top, wa_w, wa_h = 100, 200, 100, 200  # Outside Zone 1
        elif 10 <= frame_idx <= 24:
            wa_left, wa_top, wa_w, wa_h = 550, 450, 120, 300  # Inside Zone 1 (x: ~0.32, y: ~0.69)
        else:
            wa_left, wa_top, wa_w, wa_h = 100, 200, 100, 200  # Exited Zone 1

        incs = adapter.process_raw_dict({
            "class_id": 0,
            "confidence": 0.91,
            "rect_params": {"left": wa_left, "top": wa_top, "width": wa_w, "height": wa_h},
            "track_id": 101,
            "frame_index": frame_idx,
            "timestamp": ts,
        })
        if incs:
            for inc in incs:
                print(f"  Frame {frame_idx:02d} | [ALERT {inc['severity']}] Track {inc['track_id']} -> {inc['event_type']} in '{inc['zone_name']}'")
                total_incidents.extend(incs)

        # Worker B (Track 102) - Always safe
        adapter.process_raw_dict({
            "class_id": 0,
            "confidence": 0.95,
            "rect_params": {"left": 50, "top": 50, "width": 80, "height": 180},
            "track_id": 102,
            "frame_index": frame_idx,
            "timestamp": ts,
        })

        # Worker C (Untracked) - Enters on frame 15
        if frame_idx == 15:
            u_incs = adapter.process_raw_dict({
                "class_id": 0,
                "confidence": 0.86,
                "rect_params": {"left": 600, "top": 450, "width": 100, "height": 280},
                "track_id": DEEPSTREAM_UNTRACKED_ID,
                "frame_index": frame_idx,
                "timestamp": ts,
            })
            if u_incs:
                for inc in u_incs:
                    print(f"  Frame {frame_idx:02d} | [ALERT {inc['severity']}] Untracked Person -> {inc['event_type']} in '{inc['zone_name']}'")
                    total_incidents.extend(u_incs)

    print("\n" + "-" * 80)
    print(f"[RESULTS] Total Incidents Emitted: {len(total_incidents)}")
    print(f"[VERIFY] Checking persisted JSON Lines records in '{output_log}':\n")

    with open(output_log, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for i, line in enumerate(lines, 1):
        parsed = json.loads(line)
        print(f"  Record #{i}: {line}")

    print("\n" + "=" * 80)
    print("Simulation Demo Completed Successfully.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_simulation_demo()
