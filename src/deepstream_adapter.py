"""
Sentinel Vision AI - DeepStream Perception Metadata Adapter

Bridges NVIDIA DeepStream perception pipeline metadata (pyds / NvDsBatchMeta
or structured stream records) to the ZoneIntrusionMonitor.

Features:
- Filters for person class (standard COCO class_id 0).
- Normalizes DeepStream pixel rect_params [left, top, width, height] into [x_min, y_min, x_max, y_max] in [0.0, 1.0].
- Normalizes tracking IDs (handling DeepStream unassigned sentinel values).
- Robustly handles malformed metadata, corrupted types, and out-of-bounds coordinates.
- Supports both in-pipeline Python GStreamer probes (pyds) and decoupled JSON stream records.
"""

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger("SentinelVision.Adapter")

# DeepStream sentinel for unassigned tracking ID (uint64 max)
DEEPSTREAM_UNTRACKED_ID = 0xFFFFFFFFFFFFFFFF


class DeepStreamMetadataAdapter:
    def __init__(
        self,
        zone_monitor,
        person_class_id: int = 0,
        min_confidence: float = 0.40,
        default_frame_width: int = 1920,
        default_frame_height: int = 1080,
    ):
        self.zone_monitor = zone_monitor
        self.person_class_id = person_class_id
        self.min_confidence = min_confidence
        self.default_frame_width = default_frame_width
        self.default_frame_height = default_frame_height

    def normalize_rect_params(
        self,
        left: float,
        top: float,
        width: float,
        height: float,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> Optional[List[float]]:
        """
        Converts pixel-space [left, top, width, height] into normalized
        [x_min, y_min, x_max, y_max] in [0.0, 1.0] range.
        Returns None if coordinates are non-numeric or invalid.
        """
        try:
            l = float(left)
            t = float(top)
            w = float(width)
            h = float(height)
            if math.isnan(l) or math.isnan(t) or math.isnan(w) or math.isnan(h):
                return None
            if w <= 0 or h <= 0:
                return None

            fw = float(frame_width or self.default_frame_width)
            fh = float(frame_height or self.default_frame_height)
            if fw <= 0 or fh <= 0:
                fw, fh = 1.0, 1.0

            x_min = max(0.0, min(1.0, l / fw))
            y_min = max(0.0, min(1.0, t / fh))
            x_max = max(0.0, min(1.0, (l + w) / fw))
            y_max = max(0.0, min(1.0, (t + h) / fh))

            if x_min >= x_max or y_min >= y_max:
                return None

            return [x_min, y_min, x_max, y_max]
        except (ValueError, TypeError):
            return None

    def clean_track_id(self, track_id: Any) -> Optional[int]:
        """
        Normalizes tracker IDs. Returns None if unassigned or matching
        DeepStream sentinel untracked value.
        """
        if track_id is None:
            return None
        try:
            tid = int(track_id)
            if tid == DEEPSTREAM_UNTRACKED_ID or tid < 0:
                return None
            return tid
        except (ValueError, TypeError):
            return None

    def process_detection_record(
        self,
        class_id: int,
        confidence: float,
        bbox_normalized: List[float],
        track_id: Optional[int] = None,
        frame_index: Optional[int] = None,
        timestamp_epoch: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Validates and routes a single perception record to the zone monitor.
        """
        try:
            cid = int(class_id)
            conf = float(confidence)
        except (ValueError, TypeError):
            return []

        if cid != self.person_class_id:
            return []

        if math.isnan(conf) or conf < self.min_confidence or conf > 1.0:
            return []

        if not bbox_normalized or len(bbox_normalized) != 4:
            return []

        cleaned_track_id = self.clean_track_id(track_id)

        try:
            f_idx = int(frame_index) if frame_index is not None else None
            t_epoch = float(timestamp_epoch) if timestamp_epoch is not None else None
        except (ValueError, TypeError):
            f_idx = None
            t_epoch = None

        return self.zone_monitor.process_detection(
            bbox=bbox_normalized,
            confidence=conf,
            track_id=cleaned_track_id,
            frame_index=f_idx,
            timestamp_epoch=t_epoch,
        )

    def process_raw_dict(self, record: Any) -> List[Dict[str, Any]]:
        """
        Processes a dictionary containing detection fields from a message broker
        or serialized DeepStream event payload. Safely skips malformed entries.
        """
        if not isinstance(record, dict):
            return []

        try:
            class_id = record.get("class_id", record.get("classId", self.person_class_id))
            confidence = record.get("confidence", record.get("score", 0.0))
            track_id = record.get("track_id", record.get("objectId", record.get("object_id")))
            frame_index = record.get("frame_index", record.get("frame_num", record.get("frameNum")))
            timestamp = record.get("timestamp_epoch", record.get("timestamp"))

            # Bounding box handling (accepts normalized or raw rect_params)
            if "bbox" in record and isinstance(record["bbox"], (list, tuple)) and len(record["bbox"]) == 4:
                bbox_raw = [float(c) for c in record["bbox"]]
                if any(math.isnan(c) for c in bbox_raw):
                    return []
                # Check if coordinates are normalized [0, 1]
                if all(0.0 <= c <= 1.0 for c in bbox_raw):
                    bbox = bbox_raw
                else:
                    fw = float(record.get("frame_width", self.default_frame_width))
                    fh = float(record.get("frame_height", self.default_frame_height))
                    bbox = [bbox_raw[0] / fw, bbox_raw[1] / fh, bbox_raw[2] / fw, bbox_raw[3] / fh]
            elif "rect_params" in record and isinstance(record["rect_params"], dict):
                rp = record["rect_params"]
                fw = record.get("frame_width", self.default_frame_width)
                fh = record.get("frame_height", self.default_frame_height)
                bbox = self.normalize_rect_params(rp["left"], rp["top"], rp["width"], rp["height"], fw, fh)
            elif all(k in record for k in ("left", "top", "width", "height")):
                fw = record.get("frame_width", self.default_frame_width)
                fh = record.get("frame_height", self.default_frame_height)
                bbox = self.normalize_rect_params(record["left"], record["top"], record["width"], record["height"], fw, fh)
            else:
                return []

            if not bbox:
                return []

            return self.process_detection_record(
                class_id=int(class_id),
                confidence=float(confidence),
                bbox_normalized=bbox,
                track_id=track_id,
                frame_index=frame_index,
                timestamp_epoch=timestamp,
            )
        except Exception as e:
            logger.debug(f"Skipping malformed detection record: {e}")
            return []

    def process_pyds_frame_meta(self, frame_meta, pyds_module=None) -> List[Dict[str, Any]]:
        """
        Processes a single DeepStream NvDsFrameMeta inside a GStreamer pad probe.
        Iterates over frame_meta.obj_meta_list.
        Requires pyds runtime module inside DeepStream container.
        """
        all_incidents = []
        try:
            frame_index = frame_meta.frame_num
            frame_width = frame_meta.source_frame_width
            frame_height = frame_meta.source_frame_height

            obj_list = frame_meta.obj_meta_list
            while obj_list is not None:
                try:
                    obj_meta = (
                        pyds_module.NvDsObjectMeta.cast(obj_list.data)
                        if pyds_module
                        else obj_list.data
                    )
                except StopIteration:
                    break

                class_id = obj_meta.class_id
                confidence = obj_meta.confidence
                track_id = obj_meta.object_id
                rp = obj_meta.rect_params

                bbox_normalized = self.normalize_rect_params(
                    left=rp.left,
                    top=rp.top,
                    width=rp.width,
                    height=rp.height,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )

                if bbox_normalized:
                    incidents = self.process_detection_record(
                        class_id=class_id,
                        confidence=confidence,
                        bbox_normalized=bbox_normalized,
                        track_id=track_id,
                        frame_index=frame_index,
                    )
                    all_incidents.extend(incidents)

                try:
                    obj_list = obj_list.next
                except StopIteration:
                    break
        except Exception as e:
            logger.error(f"Error processing NvDsFrameMeta in pad probe: {e}")

        return all_incidents
