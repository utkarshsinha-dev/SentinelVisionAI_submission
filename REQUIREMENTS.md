# Sentinel Vision AI — Requirements Specification

This document specifies the functional, technical, and environment requirements for **Sentinel Vision AI**, developed for the **CraftifAI Physical AI Hackathon (Track A: PipeGen Perception)**.

---

## 1. Hackathon Constraints & Ground Rules

1. **Mandatory PipeGen Perception Workflow**:
   The perception pipeline must strictly follow the required 5-stage lifecycle:
   $$\text{Requirement} \longrightarrow \text{Pipeline IR} \longrightarrow \text{Model} \longrightarrow \text{Compiler} \longrightarrow \text{Run}$$
2. **No Fallback / Mock Substitutes**:
   A standalone Python/OpenCV detector script cannot be used as a substitute for CraftifAI PipeGen and NVIDIA DeepStream.
3. **Evidence-Based API Integration**:
   PipeGen commands, Pipeline IR schemas, and compiler parameters must not be invented or assumed. They will be derived directly by inspecting the installed CraftifAI PipeGen v0.5.1 package and documentation.
4. **Hardware Acceleration**:
   Inference and video decoding must utilize the host's NVIDIA GeForce RTX 4060 GPU via NVIDIA Container Toolkit.
5. **Actionable Deliverables**:
   The system must output structured incident records (JSON) and concrete performance benchmarks (TensorRT speed report).

---

## 2. Functional Requirements (FR)

### FR-1: Video Ingestion
- **FR-1.1**: Support ingestion of local video files (e.g., MP4/H.264, H.265, or MKV) representing workplace or industrial surveillance footage.
- **FR-1.2**: Support configurable input video path via a configuration file or command-line parameter.

### FR-2: Person Detection & Tracking
- **FR-2.1**: Accurately detect human personnel in each video frame with bounding box coordinates and detection confidence scores.
- **FR-2.2**: Maintain consistent object tracking IDs (`track_id`) across frames to monitor continuous presence and trajectory.

### FR-3: Configurable Restricted Zones (ROI)
- **FR-3.1**: Support defining one or more restricted zones as 2D arbitrary polygons specified by an ordered list of coordinate vertices $[(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)]$.
- **FR-3.2**: Allow zone configuration in a human-readable file (`configs/restricted_zone.json`) with attributes including `zone_id`, `name`, `severity_level`, and `vertices`.

### FR-4: Spatial Intrusion Logic
- **FR-4.1**: Compute the person's ground contact point (defined as the bottom-center of the bounding box: $(x_{\text{center}}, y_{\text{max}})$).
- **FR-4.2**: Determine whether the ground contact point lies within any defined restricted polygon using standard point-in-polygon (ray casting) testing.
- **FR-4.3**: Trigger an intrusion event when a tracked person enters or remains inside the restricted zone.

### FR-5: Actionable Structured Incident Output
- **FR-5.1**: For every intrusion event, generate a structured JSON incident record containing:
  - `incident_id`: Unique UUID or sequential identifier.
  - `timestamp_iso`: UTC ISO 8601 timestamp.
  - `frame_index`: Video frame index of occurrence.
  - `track_id`: Unique tracking ID of the intruding person.
  - `zone_id`: Identifier of the violated zone.
  - `zone_name`: Human-readable zone designation (e.g., `"Heavy Machinery Cell A"`).
  - `severity`: Alert severity level (`"WARNING"`, `"CRITICAL"`).
  - `confidence`: Model detection confidence score ($0.0 - 1.0$).
  - `bbox`: Person bounding box `[x_min, y_min, x_max, y_max]`.
  - `ground_point`: Computed footprint coordinate `[x, y]`.
- **FR-5.2**: Persist incident records to a structured JSON Lines file (`data/output/incident_records.jsonl`) for downstream alerting or audit logging.

---

## 3. Technical Requirements (TR)

### TR-1: PipeGen Pipeline IR Conformance
- **TR-1.1**: The perception pipeline must be defined as an official PipeGen Pipeline IR file (`artifacts/ir/pipeline_ir.yaml` or `.json`).
- **TR-1.2**: The IR must define video source decoding, model inference node, tracker node, and metadata output sink.
- **TR-1.3**: The Pipeline IR must pass PipeGen's IR validation tool.

### TR-2: Model Selection & TensorRT Compilation
- **TR-2.1**: The person detector model must be compiled using PipeGen's compiler targeting the NVIDIA RTX 4060 GPU (Ada Lovelace architecture, SM 8.9).
- **TR-2.2**: The compilation process must yield a serialized TensorRT `.engine` file.
- **TR-2.3**: Generate a formal speed/benchmark report (`artifacts/benchmarks/speed_report.md`) documenting:
  - Model precision (e.g., FP16 or INT8)
  - Average inference latency (ms per frame)
  - Processing throughput (Frames Per Second)
  - Peak GPU memory (VRAM) footprint

### TR-3: NVIDIA DeepStream Execution
- **TR-3.1**: The compiled pipeline must execute inside the PipeGen-managed NVIDIA DeepStream container.
- **TR-3.2**: Utilize NVIDIA hardware acceleration for video decoding (NVDEC) and inference execution.

### TR-4: Software Modularity & Clean Architecture
- **TR-4.1**: Workspace must maintain clear separation: configuration (`configs/`), data (`data/`), artifacts (`artifacts/`), and source logic (`src/`).
- **TR-4.2**: Execution scripts must be deterministic, non-destructive, and reproducible.

---

## 4. Hardware & Environment Specifications

| Component | Specification |
| :--- | :--- |
| **Host Operating System** | Windows 11 (64-bit) |
| **Subsystem** | WSL2 (Ubuntu 24.04 LTS) |
| **CPU** | Multi-core x86_64 host processor |
| **GPU** | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6 VRAM) |
| **NVIDIA Compute Capability** | 8.9 (Ada Lovelace) |
| **Container Engine** | Docker CE + NVIDIA Container Toolkit (`nvidia-docker2` / `nvidia-ctk`) |
| **Frameworks** | CraftifAI PipeGen v0.5.1, NVIDIA DeepStream SDK |
| **Test Input** | Local high-definition workplace video file (no external camera required) |
