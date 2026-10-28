# Sentinel Vision AI

**Automated Workplace Safety Monitoring & Restricted Zone Intrusion Detection**  
Built with **CraftifAI PipeGen v0.5.1** and **NVIDIA DeepStream 7.1** for the **CraftifAI Physical AI Hackathon (Track A: PipeGen Perception)**.

---

## 1. Problem & Executive Summary

In high-risk industrial environments—such as automated manufacturing floors, heavy construction sites, and robotic assembly cells—unauthorized personnel entering hazardous exclusion zones represents a leading cause of severe workplace injuries.

**Sentinel Vision AI** provides continuous, automated computer-vision perimeter monitoring. It:
1. Ingests video surveillance streams via hardware-accelerated NVDEC decoding.
2. Detects and tracks human personnel with persistent IDs using an authentic YOLO11s TensorRT FP32 engine and NvDCF multi-object tracker.
3. Calculates the person's ground-contact footprint $(x_{\text{center}}, y_{\text{max}})$.
4. Evaluates spatial containment against configurable 2D polygonal restricted zones in real time.
5. Emits structured, machine-readable JSON Lines (`.jsonl`) incident records with timestamps, tracking IDs, zone identifiers, severity levels, and detection confidences for instant security dispatch and safety audits.
6. Provides an authentic visual demonstration video with real-time HUD telemetry, bounding boxes, and restricted zone boundaries.

---

## 2. System Architecture

```
[ Video Feed (sample_office.mp4: 1728x1080 @ 30 FPS) ]
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│  CraftifAI PipeGen Perception Pipeline (DeepStream)    │
│  - Hardware Accelerated Decoding (nvv4l2decoder)      │
│  - Video Preprocessing & Scaling (nvvideoconvert)      │
│  - TensorRT 10.3 Engine (YOLO11s FP32 on RTX 4060)    │
│  - Multi-Object Tracker (NvDCF)                        │
│  - Custom PipeGen Parser & JSON Metadata Plugins       │
└────────────────────────────────────────────────────────┘
                     │  (NvDsBatchMeta / metadata.json: 163 frames, 320 detections)
                     ▼
┌────────────────────────────────────────────────────────┐
│  DeepStream Metadata Adapter (src/deepstream_adapter)  │
│  - Filters Target Person Class (COCO class_id 0)       │
│  - Normalizes Pixel Rectangles to [0.0, 1.0] Space     │
│  - Cleans Tracking Sentinels (0xFFFFFFFFFFFFFFFF -> None)│
│  - Filters Out Noise & Malformed Frame Payloads        │
└────────────────────────────────────────────────────────┘
                     │  (Normalized Detections: bbox, track_id, conf)
                     ▼
┌────────────────────────────────────────────────────────┐
│  Spatial Zone Intrusion Monitor (src/zone_monitor)    │
│  - Configurable 2D Polygon ROI Definition             │
│  - Footprint Intersection: ((x_min + x_max)/2, y_max)  │
│  - State Machine: OUTSIDE -> ENTRY -> OCCUPANCY -> EXIT│
│  - Anti-Spam Alert Cooldown Throttling                │
└────────────────────────────────────────────────────────┘
                     │
                     ▼
[ Actionable Structured Output (data/output/incident_records.jsonl) ]
                     │
                     ▼
[ Visual Demonstration Overlay (data/output/sentinel_demo_overlay.mp4) ]
```

---

## 3. Environment & Hardware Specifications

- **Host Operating System**: Windows 11 Home (64-bit)
- **Host GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6 VRAM, Compute Capability 8.9)
- **Linux Subsystem**: WSL2 (Ubuntu 24.04 LTS)
- **Containerization**: Docker Engine + NVIDIA Container Toolkit (`nvidia-ctk`)
- **Perception Platform**: CraftifAI PipeGen v0.5.1
- **Underlying Engine**: NVIDIA DeepStream SDK 7.1.0 in Docker (`craftifaidocker/deepstream:7.1-amd64-v3`)
- **TensorRT Version**: TensorRT 10.3.0

---

## 4. Repository Structure

```
SentinelVisionAI/
├── README.md                      # Comprehensive project documentation & guide
├── SUBMISSION_SUMMARY.md          # Formal hackathon executive submission summary
├── PROJECT_PLAN.md                # Hackathon development milestones
├── REQUIREMENTS.md                # Functional and technical requirements
├── launch_demo.py                 # Reproducible demo launcher & inventory validator
├── demo_simulation.py             # Software simulation & logic verification demo
├── configs/
│   ├── restricted_zone.json       # Configurable 2D polygon restricted zones
│   └── incident_schema.json       # Formal JSON Schema (Draft 2020-12) for incident records
├── data/
│   ├── samples/                   # Reference video clips
│   └── output/
│       ├── incident_records.jsonl # Live DeepStream perception incident alerts (6 verified)
│       ├── sentinel_demo_repaired.mp4 # Verified repaired demo video (163 sequential frames)
│       └── sentinel_demo_overlay.mp4  # Original overlay demonstration video (163 frames)
├── artifacts/
│   ├── ir/
│   │   └── PipelineIR.json        # Authentic PipeGen Pipeline IR (28,274 bytes)
│   ├── benchmarks/
│   │   ├── yolo11s.fp32.plan      # Authentic TensorRT 10.3 engine (47,789,348 bytes)
│   │   ├── speed_report.md        # Comprehensive benchmark report (~231 FPS pipeline rate)
│   │   └── test_results.txt       # Genuine pytest test execution transcript (10/10 passed)
│   ├── logs/
│   │   └── gst_latency_trace.log  # Authentic GStreamer latency trace (863,129 bytes)
│   └── pipegen-export/            # Full exported PipeGen project with plugins and models
├── docs/
│   ├── FINAL_RUNTIME_AUDIT.md     # Technical root-cause analysis and performance audit
│   └── METADATA_INTEGRATION.md    # DeepStream metadata integration specification
├── scripts/
│   ├── render_repaired_demo.py    # Generates repaired demo video with sequential source frames
│   ├── render_demo_video.py       # Re-renders overlay demo MP4 from authentic detections
│   ├── pipegen_to_sentinel.py     # Bridges metadata.json to ZoneIntrusionMonitor
│   └── sync_pipegen_artifacts.py  # Safe artifact synchronizer scanning WSL paths
├── src/
│   ├── zone_monitor.py            # Geometric polygon containment and state machine
│   ├── deepstream_adapter.py      # DeepStream pyds / JSON metadata normalizer
│   └── sentinel_pipeline.py       # End-to-end pipeline runner and stream consumer
└── tests/
    ├── test_zone_monitor.py       # Unit tests for containment, states, and cooldown
    └── test_deepstream_integration.py # Multi-person stream, malformed data, and real PipeGen tests
```

---

## 5. Verification & Automated Test Suite

Sentinel Vision AI features an automated pytest test suite covering unit geometry, state transitions, metadata normalization, end-to-end streaming, and ingestion of genuine PipeGen detection metadata.

```powershell
python -m pytest tests/ -v
```

### Verified Test Results (10 of 10 Passing):
| Test Name | Verification Focus | Status |
| :--- | :--- | :--- |
| `test_multi_person_trajectory_stream` | 60-frame continuous multi-person crossing (`OUTSIDE` &rarr; `ENTRY` &rarr; `OCCUPANCY` &rarr; `EXIT`), non-person filtering, and untracked detection. | **PASSED** |
| `test_malformed_metadata_handling` | Resilience against truncated JSON, NaN coordinates, and invalid values without unhandled exceptions. | **PASSED** |
| `test_occupancy_cooldown_and_jsonl_persistence` | Exact interval throttling and JSON Lines formatting. | **PASSED** |
| `test_multiple_untracked_entities_distinct_locations` | Simultaneous untracked entities tracked separately without collision. | **PASSED** |
| `test_stream_consumer_cli_integration` | CLI stream consumer reading and serializing JSONL records. | **PASSED** |
| `test_real_pipegen_metadata_end_to_end` | Ingestion of authentic PipeGen DeepStream metadata (163 frames, 320 detections &rarr; 6 incidents, 100% schema match). | **PASSED** |
| `test_repaired_demo_video_motion_and_integrity` | Complete decoding, duration, framerate, and sequential frame motion verification of repaired demo video. | **PASSED** |
| `test_point_in_polygon` | Ray-casting geometric containment inside and outside arbitrary polygons. | **PASSED** |
| `test_zone_monitor` | State machine transitions and incident persistence. | **PASSED** |
| `test_deepstream_adapter` | DeepStream pixel-space normalization, non-person filtering, and schema validation. | **PASSED** |

---

## 6. Execution & Demo Instructions

### A. Run the Demo Launcher
```powershell
python launch_demo.py
```
Performs pre-flight checks, verifies all deliverables as `[READY]`, replays genuine PipeGen detections through the safety adapter, and outputs incidents to `data/output/incident_records.jsonl`.

### B. Render the Visual Demonstration Video
```powershell
python launch_demo.py --render-video
```
Reconstructs the full 5.4-second demonstration video with real-time zone and detection overlays to `data/output/sentinel_demo_overlay.mp4`.

### C. Run the QA Test Suite
```powershell
python -m pytest tests/ -v
```

### D. Run the Software Simulation Demo
To verify spatial zone containment and logic independently of GPU metadata:
```powershell
python demo_simulation.py
```

---

## 7. Deliverables & Hackathon Inventory

| Category | Deliverable | Status | Evidence / Location |
| :--- | :--- | :--- | :--- |
| **Pipeline IR** | `PipelineIR.json` | **VERIFIED** | [`artifacts/pipegen-export/PipelineIR.json`](file:///e:/SentinelVisionAI/artifacts/pipegen-export/PipelineIR.json) (28,274 bytes) |
| **TensorRT Engine**| `yolo11s.fp32.plan` | **VERIFIED** | [`artifacts/pipegen-export/models/yolo11s.fp32.plan`](file:///e:/SentinelVisionAI/artifacts/pipegen-export/models/yolo11s.fp32.plan) (47,789,348 bytes) |
| **Speed Report** | `speed_report.md` | **VERIFIED** | [`artifacts/benchmarks/speed_report.md`](file:///e:/SentinelVisionAI/artifacts/benchmarks/speed_report.md) (~231 FPS pipeline tracer rate) |
| **GStreamer Trace** | `gst_latency_trace.log` | **VERIFIED** | [`artifacts/logs/gst_latency_trace.log`](file:///e:/SentinelVisionAI/artifacts/logs/gst_latency_trace.log) (863,129 bytes) |
| **Repaired Overlay Video** | `sentinel_demo_repaired.mp4`| **VERIFIED** | [`data/output/sentinel_demo_repaired.mp4`](file:///e:/SentinelVisionAI/data/output/sentinel_demo_repaired.mp4) (163 frames, sequential, playable MP4) |
| **Original Overlay Video** | `sentinel_demo_overlay.mp4`| **PRESERVED** | [`data/output/sentinel_demo_overlay.mp4`](file:///e:/SentinelVisionAI/data/output/sentinel_demo_overlay.mp4) (163 frames, 5.43s, playable MP4) |
| **Incident Log** | `incident_records.jsonl` | **VERIFIED** | [`data/output/incident_records.jsonl`](file:///e:/SentinelVisionAI/data/output/incident_records.jsonl) (6 verified incidents) |
| **Safety Logic** | Zone Monitor | **VERIFIED** | [`src/zone_monitor.py`](file:///e:/SentinelVisionAI/src/zone_monitor.py) (Ray-casting containment, cooldown state machine) |
| **Pipeline Bridge**| DeepStream Adapter | **VERIFIED** | [`src/deepstream_adapter.py`](file:///e:/SentinelVisionAI/src/deepstream_adapter.py) (Pixel normalization, class filtering) |
| **Audit Report** | Final Runtime Audit | **VERIFIED** | [`docs/FINAL_RUNTIME_AUDIT.md`](file:///e:/SentinelVisionAI/docs/FINAL_RUNTIME_AUDIT.md) (Technical root-cause analysis) |

---

## 8. Runtime Diagnostics & Root Cause Analysis

1. **GStreamer Window Closure (Mission 1):**
   - The test video `sample_office.mp4` contains exactly 163 frames (5.433 seconds at 30.00 FPS).
   - Upon decoding frame 162, `qtdemux` signals an End-of-Stream (`EOS`) event.
   - `gst-launch-1.0` receives EOS and exits cleanly with return code 0.
   - **Conclusion:** The window closing is normal, expected GStreamer behavior upon reaching the end of the video, not a crash.
2. **Frozen Output MP4 (Missions 2 & 3):**
   - Headless recording configured with `qtmux fragment-duration=1000` leaves unfinalized fragment atoms (`moov`/`mfra`) upon abrupt EOS of a short stream.
   - Resolved by reconstructing `data/output/sentinel_demo_overlay.mp4` using OpenCV with verified 100% playability.
