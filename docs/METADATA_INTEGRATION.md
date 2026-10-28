# Sentinel Vision AI — Metadata Integration Architecture

This document specifies how detection and tracking metadata produced by the **CraftifAI PipeGen** DeepStream perception pipeline connects to the **Sentinel Vision AI** restricted zone incident alerting logic.

---

## 1. Ground Rules & Non-Assumptions

1. **No Invented APIs**:
   PipeGen compiles the perception pipeline into an NVIDIA DeepStream implementation (as evidenced by PipeGen's generated artifact list: `pipeline.py`, `deepstream_app_config.txt`, `config_infer_primary_yolov8n.txt`, `config_tracker_nvdcf.yml`). We do not guess proprietary PipeGen Python wrappers. Instead, integration connects directly to the standard DeepStream metadata structures (`pyds`) exposed in the generated pipeline.
2. **Zero Mock Fallbacks**:
   The incident monitor will only receive events sourced from the running GPU pipeline.

---

## 2. NVIDIA DeepStream Perception Metadata Model

In DeepStream pipelines, frame and object metadata traverse the GStreamer pipeline attached to each `GstBuffer` via NVIDIA's `NvDsBatchMeta` structure.

```
GstBuffer
   │
   ▼
[ NvDsBatchMeta ]
   │
   └──► [ NvDsFrameMeta ] (frame_num, source_id, source_frame_width, source_frame_height)
           │
           └──► [ NvDsObjectMeta ] (Iterated per detected entity)
                   ├── class_id    : integer (0 for 'person' in standard COCO/YOLO)
                   ├── object_id   : uint64 (Persistent tracking ID assigned by NvDCF)
                   ├── confidence  : float (Inference detection confidence)
                   └── rect_params : [left, top, width, height] in pixel coordinates
```

### Extracted Object Attributes

| DeepStream Field | Sentinel Variable | Semantic Role |
| :--- | :--- | :--- |
| `obj_meta.class_id` | `class_id` | Filters for person detections (COCO class `0`). |
| `obj_meta.object_id` | `track_id` | Persistent person tracker ID across frames. |
| `obj_meta.confidence` | `confidence` | Model confidence score ($0.0 - 1.0$). |
| `obj_meta.rect_params` | `[x1, y1, x2, y2]` | Bounding box spatial coordinates. |
| `frame_meta.frame_num` | `frame_index` | Frame sequence index. |

---

## 3. Spatial Intrusion & Ground Contact Geometry

In surveillance and industrial safety monitoring, projecting a person's 3D ground location is standardly approximated by the **ground contact point** (footprint):

$$P_{\text{ground}} = \left( \frac{x_{\min} + x_{\max}}{2}, \; y_{\max} \right)$$

Where:
- $x_{\min} = \frac{\text{rect\_params.left}}{\text{frame\_width}}$
- $y_{\min} = \frac{\text{rect\_params.top}}{\text{frame\_height}}$
- $x_{\max} = \frac{\text{rect\_params.left} + \text{rect\_params.width}}{\text{frame\_width}}$
- $y_{\max} = \frac{\text{rect\_params.top} + \text{rect\_params.height}}{\text{frame\_height}}$

All coordinates are normalized to the interval $[0.0, 1.0]$, ensuring zone definitions configured in `configs/restricted_zone.json` remain resolution-invariant.

---

## 4. Intrusion Detection & Anti-Spam Cooldown Logic

To avoid repetitive alert flooding for every individual frame an authorized worker or intruder is inside a zone:

```
For each frame:
  For each detected person (track_id):
    1. Calculate P_ground = ((x_min + x_max)/2, y_max)
    2. For each zone in configs/restricted_zone.json:
         is_inside = point_in_polygon(P_ground, zone.polygon)
         
         if is_inside:
           current_state = tracker_state.get(track_id, OUTSIDE)
           last_alert_time = last_alert_time.get((track_id, zone_id), 0)
           
           if current_state == OUTSIDE:
             Emit Incident (event_type="ZONE_INTRUSION_ENTRY", severity=zone.severity)
             tracker_state[track_id] = INSIDE
             last_alert_time[(track_id, zone_id)] = current_time
             
           elif (current_time - last_alert_time) > zone.cooldown_seconds:
             Emit Incident (event_type="ZONE_INTRUSION_OCCUPANCY", severity=zone.severity)
             last_alert_time[(track_id, zone_id)] = current_time
             
         else:
           if tracker_state.get(track_id) == INSIDE:
             Emit Incident (event_type="ZONE_INTRUSION_EXIT", severity="INFO")
             tracker_state[track_id] = OUTSIDE
```

---

## 5. Integration Attachment Point

Depending on PipeGen's generated code layout (to be confirmed upon compile):

1. **Option A (DeepStream Python Probe Hook)**:
   In `pipeline.py`, attach a pad probe (`gst_pad_add_probe`) to the output pad of `nvtracker` or `nvdsosd`. The probe function invokes Sentinel Vision AI's `ZoneIntrusionMonitor` passing the frame metadata.
2. **Option B (DeepStream Message Converter / Output Log)**:
   If PipeGen configures an out-of-the-box metadata sink (e.g., via `msgconv_config.txt` or JSON stream output as shown in PipeGen's VMS console), Sentinel's monitor streams the raw detection records and performs the polygon containment test.

Both options consume the exact same underlying `NvDsObjectMeta` fields, preserving strict compliance without modifying proprietary internals.
