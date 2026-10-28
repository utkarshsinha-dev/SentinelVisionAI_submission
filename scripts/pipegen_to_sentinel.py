import argparse
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.zone_monitor import ZoneIntrusionMonitor
from src.deepstream_adapter import DeepStreamMetadataAdapter

parser = argparse.ArgumentParser(description="PipeGen Metadata to Sentinel Safety Monitor")
parser.add_argument("--metadata", default=str(root / "artifacts/pipegen-export/output/metadata.json"), help="PipeGen metadata JSON path")
parser.add_argument("--config", default=str(root / "configs/restricted_zone.json"), help="Restricted zone config path")
parser.add_argument("--output", default=str(root / "data/output/incident_records.jsonl"), help="Incident records output JSONL")
args, _ = parser.parse_known_args()

metadata_path = Path(args.metadata)
config_path = Path(args.config)
output_path = Path(args.output)
pipegen_copy_path = root / "data/output/pipegen_incidents.jsonl"

with metadata_path.open(encoding="utf-8") as f:
    metadata = json.load(f)

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text("", encoding="utf-8")

monitor = ZoneIntrusionMonitor(
    config_path=str(config_path),
    output_log_path=str(output_path),
)
adapter = DeepStreamMetadataAdapter(
    zone_monitor=monitor,
    min_confidence=0.40,
    default_frame_width=1728,
    default_frame_height=1080,
)

detections = 0
incidents = []

for frame in metadata.get("frames", []):
    width = frame.get("width", 1728)
    height = frame.get("height", 1080)

    for obj in frame.get("objects", []):
        bbox = obj.get("bbox", {})
        if not isinstance(bbox, dict):
            continue

        detections += 1
        record = {
            "class_id": obj.get("class_id", 0),
            "confidence": obj.get("confidence", 0.0),
            "track_id": obj.get("id"),
            "frame_index": frame.get("frame"),
            "timestamp_epoch": (
                frame.get("pts_ns", 0) / 1_000_000_000
            ),
            "frame_width": width,
            "frame_height": height,
            "rect_params": {
                "left": bbox.get("left"),
                "top": bbox.get("top"),
                "width": bbox.get("width"),
                "height": bbox.get("height"),
            },
        }

        incidents.extend(adapter.process_raw_dict(record))

if output_path.exists() and output_path != pipegen_copy_path:
    pipegen_copy_path.write_bytes(output_path.read_bytes())
    print(f"Also synchronized to: {pipegen_copy_path}")

from collections import Counter
counts = Counter(i["event_type"] for i in incidents)

print("Sentinel Vision AI — PipeGen Detection Bridge")
print(f"Frames: {len(metadata.get('frames', []))}")
print(f"Detection records read: {detections}")
print(f"Incidents generated: {len(incidents)}")
print(f"Event breakdown: {dict(counts)}")
print(f"Incident log: {output_path}")

if not incidents:
    print(
        "No zone incidents generated. This can mean the detections "
        "did not enter the configured polygons; it does not prove "
        "that the zone monitor is broken."
    )
