"""
Sentinel Vision AI - Restricted Zone Intrusion Monitor

Evaluates person bounding boxes and tracking IDs against configurable restricted
zones, emitting structured JSON incident records upon zone breaches.
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any


def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """
    Ray-casting algorithm to test whether a 2D point (x, y) is inside a polygon.
    All coordinates should be in the same space (e.g., normalized [0.0, 1.0]).
    """
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


class ZoneIntrusionMonitor:
    def __init__(
        self,
        config_path: str,
        output_log_path: str = "data/output/incident_records.jsonl",
    ):
        self.config_path = config_path
        self.output_log_path = output_log_path
        self.zones: List[Dict[str, Any]] = []
        self.tracker_states: Dict[Tuple[Any, str], str] = {}  # (track_id, zone_id) -> "INSIDE" | "OUTSIDE"
        self.last_alert_time: Dict[Tuple[Any, str], float] = {}  # (track_id, zone_id) -> epoch timestamp

        self._load_config()
        os.makedirs(os.path.dirname(self.output_log_path), exist_ok=True)

    def _load_config(self) -> None:
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Zone config not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.zones = data.get("zones", [])

    @staticmethod
    def compute_ground_contact(bbox: List[float]) -> Tuple[float, float]:
        """
        Calculates ground contact point (bottom-center of bounding box).
        bbox: [x_min, y_min, x_max, y_max] in normalized coordinates [0.0, 1.0].
        """
        x_min, y_min, x_max, y_max = bbox
        return ((x_min + x_max) / 2.0, y_max)

    def process_detection(
        self,
        bbox: List[float],
        confidence: float,
        track_id: Optional[int] = None,
        frame_index: Optional[int] = None,
        timestamp_epoch: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Processes a single person detection against all configured zones.
        Returns a list of generated incident records (if any).
        """
        current_time = timestamp_epoch or time.time()
        ground_point = self.compute_ground_contact(bbox)
        generated_incidents = []

        entity_id = track_id if track_id is not None else f"untracked_{bbox[0]:.2f}_{bbox[1]:.2f}"

        for zone in self.zones:
            zone_id = zone["zone_id"]
            polygon = zone["polygon"]
            severity = zone.get("severity", "WARNING")
            zone_name = zone.get("zone_name", zone_id)
            cooldown = zone.get("cooldown_seconds", 2.0)

            state_key = (entity_id, zone_id)
            is_inside = point_in_polygon(ground_point, polygon)
            previous_state = self.tracker_states.get(state_key, "OUTSIDE")

            if is_inside:
                last_alert = self.last_alert_time.get(state_key, 0.0)
                event_type = None

                if previous_state == "OUTSIDE":
                    event_type = "ZONE_INTRUSION_ENTRY"
                elif (current_time - last_alert) >= cooldown:
                    event_type = "ZONE_INTRUSION_OCCUPANCY"

                if event_type:
                    incident = {
                        "incident_id": str(uuid.uuid4()),
                        "timestamp_iso": datetime.fromtimestamp(current_time, tz=timezone.utc).isoformat(),
                        "frame_index": frame_index,
                        "event_type": event_type,
                        "zone_id": zone_id,
                        "zone_name": zone_name,
                        "severity": severity,
                        "track_id": track_id,
                        "confidence": round(float(confidence), 4),
                        "bbox": [round(float(c), 4) for c in bbox],
                        "ground_contact_point": [round(ground_point[0], 4), round(ground_point[1], 4)],
                    }
                    self._record_incident(incident)
                    generated_incidents.append(incident)
                    self.last_alert_time[state_key] = current_time

                self.tracker_states[state_key] = "INSIDE"

            else:
                if previous_state == "INSIDE":
                    # Entity exited the zone
                    exit_incident = {
                        "incident_id": str(uuid.uuid4()),
                        "timestamp_iso": datetime.fromtimestamp(current_time, tz=timezone.utc).isoformat(),
                        "frame_index": frame_index,
                        "event_type": "ZONE_INTRUSION_EXIT",
                        "zone_id": zone_id,
                        "zone_name": zone_name,
                        "severity": "INFO",
                        "track_id": track_id,
                        "confidence": round(float(confidence), 4),
                        "bbox": [round(float(c), 4) for c in bbox],
                        "ground_contact_point": [round(ground_point[0], 4), round(ground_point[1], 4)],
                    }
                    self._record_incident(exit_incident)
                    generated_incidents.append(exit_incident)
                    self.tracker_states[state_key] = "OUTSIDE"

        return generated_incidents

    def _record_incident(self, incident: Dict[str, Any]) -> None:
        with open(self.output_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(incident) + "\n")
