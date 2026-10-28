"""
Sentinel Vision AI - End-to-End Pipeline Runner & Metadata Stream Consumer

Connects the real perception pipeline (PipeGen / DeepStream) to the
ZoneIntrusionMonitor, ensuring live person tracks are evaluated against
configured restricted zones and actionable incidents are logged.

Supports:
- Real-time GStreamer buffer probe callback (in-container pyds execution)
- Stream consumption from file or stdin (pipe: ./run.sh | python src/sentinel_pipeline.py)
- Generator / iterable interface for programmatic and unit-test execution
"""

import argparse
import json
import os
import sys
import time
from typing import Iterable, Optional, Tuple, List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.zone_monitor import ZoneIntrusionMonitor
from src.deepstream_adapter import DeepStreamMetadataAdapter


def consume_stream_iterable(
    lines: Iterable[str],
    adapter: DeepStreamMetadataAdapter,
    verbose: bool = True,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Processes an iterable of JSON lines through the DeepStream adapter and zone monitor.
    Returns (total_detections_processed, total_incidents_generated, list_of_incidents).
    """
    total_detections = 0
    all_generated_incidents = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        incidents = adapter.process_raw_dict(record)
        total_detections += 1
        if incidents:
            all_generated_incidents.extend(incidents)
            if verbose:
                for inc in incidents:
                    print(
                        f"  [ALERT {inc['severity']}] Zone: {inc['zone_name']} "
                        f"({inc['zone_id']}) | Track: {inc['track_id']} | "
                        f"Event: {inc['event_type']} | Conf: {inc['confidence']:.2f}"
                    )

    return total_detections, len(all_generated_incidents), all_generated_incidents


def run_stream_consumer(
    zone_config: str = "configs/restricted_zone.json",
    output_log: str = "data/output/incident_records.jsonl",
    input_stream_file: Optional[str] = None,
    default_width: int = 1920,
    default_height: int = 1080,
    verbose: bool = True,
) -> Tuple[int, int]:
    """
    End-to-end stream consumer reading from a file, stdin, or named pipe.
    """
    if verbose:
        print("=" * 70)
        print("SENTINEL VISION AI — Workplace Safety Monitoring System")
        print("=" * 70)
        print(f"Loading Restricted Zones from: {zone_config}")
        print(f"Persisting Incident Records to: {output_log}")

    monitor = ZoneIntrusionMonitor(config_path=zone_config, output_log_path=output_log)
    adapter = DeepStreamMetadataAdapter(
        zone_monitor=monitor,
        default_frame_width=default_width,
        default_frame_height=default_height,
    )

    # 1. Reading from Stdin (pipe mode)
    if input_stream_file == "-" or (input_stream_file is None and not sys.stdin.isatty()):
        if verbose:
            print("Consuming live detection stream from STDIN...")
        total_det, total_inc, _ = consume_stream_iterable(sys.stdin, adapter, verbose=verbose)

    # 2. Reading from specified file
    elif input_stream_file and os.path.exists(input_stream_file):
        if verbose:
            print(f"Consuming detection stream from file: {input_stream_file}")
        with open(input_stream_file, "r", encoding="utf-8") as f:
            total_det, total_inc, _ = consume_stream_iterable(f, adapter, verbose=verbose)

    # 3. No stream provided
    else:
        if verbose:
            print(f"[INFO] No active stream provided. Pass --stream <file.jsonl> or pipe data via stdin.")
        total_det, total_inc = 0, 0

    if verbose:
        print("-" * 70)
        print(f"Stream Consumer Finished. Processed Detections: {total_det}, Generated Incidents: {total_inc}")

    return total_det, total_inc


def attach_deepstream_probe(pad, info, adapter: DeepStreamMetadataAdapter, pyds_module):
    """
    GStreamer pad probe callback function to be attached directly to DeepStream's
    tracker or OSD sink pad in pipeline.py:
        pad.add_probe(Gst.PadProbeType.BUFFER, attach_deepstream_probe, (adapter, pyds))
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return 0  # GST_PAD_PROBE_OK

    batch_meta = pyds_module.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    frame_list = batch_meta.frame_meta_list

    while frame_list is not None:
        try:
            frame_meta = pyds_module.NvDsFrameMeta.cast(frame_list.data)
        except StopIteration:
            break

        incidents = adapter.process_pyds_frame_meta(frame_meta, pyds_module)
        for inc in incidents:
            print(
                f"[LIVE ALERT {inc['severity']}] Breached Zone: {inc['zone_name']} | "
                f"Track ID: {inc['track_id']} | Event: {inc['event_type']}"
            )

        try:
            frame_list = frame_list.next
        except StopIteration:
            break

    return 0  # GST_PAD_PROBE_OK


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentinel Vision AI Pipeline Runner")
    parser.add_argument("--config", default="configs/restricted_zone.json", help="Path to restricted zone configuration")
    parser.add_argument("--output", default="data/output/incident_records.jsonl", help="Path to incident records output")
    parser.add_argument("--stream", default=None, help="Path to input detection stream (JSON lines, or '-' for stdin)")
    parser.add_argument("--width", type=int, default=1920, help="Default frame width")
    parser.add_argument("--height", type=int, default=1080, help="Default frame height")
    args = parser.parse_args()

    run_stream_consumer(
        zone_config=args.config,
        output_log=args.output,
        input_stream_file=args.stream,
        default_width=args.width,
        default_height=args.height,
    )
