# Sentinel Vision AI — Project Plan & Milestones

This document defines the sequential development milestones for Sentinel Vision AI. Each milestone is scoped to be small, incremental, and strictly verifiable against real system commands before moving forward.

---

## Milestone Overview

```
Phase 0: Environment & PipeGen Inspection
     │
     ▼
Phase 1: Input Data & Zone Configuration Setup
     │
     ▼
Phase 2: PipeGen Requirement & Pipeline IR Specification
     │
     ▼
Phase 3: Model Selection & TensorRT Compilation
     │
     ▼
Phase 4: DeepStream Pipeline Execution via PipeGen
     │
     ▼
Phase 5: Zone Intrusion Detection & Actionable JSON Alerting
     │
     ▼
Phase 6: End-to-End Prototype Validation & Deliverable Packaging
```

---

## Phase 0: Environment & PipeGen Interface Inspection (Zero-Assumption)

> **Objective**: Inspect the real installed PipeGen v0.5.1 environment, documentation, CLI tools, and container runtime without guessing or inventing APIs.

### Steps
1. Verify WSL2 Ubuntu 24.04 environment, Docker daemon status, and NVIDIA GPU container passthrough (`nvidia-smi` inside Docker).
2. Check completion status of the DeepStream container download in PipeGen.
3. Inspect PipeGen v0.5.1 installation:
   - Run `pipegen --help` (or package equivalent) to discover available subcommands, IR validators, and compilers.
   - Inspect installed Python package modules, schemas, and bundled documentation/examples.
4. Document the discovered PipeGen schema, CLI syntax, and IR specifications.

### Verification Criteria
- [ ] `nvidia-smi` successfully displays RTX 4060 inside an NVIDIA container.
- [ ] PipeGen container download confirmed complete.
- [ ] PipeGen CLI / SDK entry points, subcommands, and Pipeline IR format documented from source evidence.

---

## Phase 1: Input Data & Zone Configuration Setup

> **Objective**: Prepare a lightweight sample video and define the restricted zone coordinates.

### Steps
1. Place a lightweight, standard test video (e.g., CCTV or workplace scene showing pedestrians/workers walking across a marked area) into `data/samples/`.
2. Define a restricted zone specification in a clean JSON format under `configs/restricted_zone.json` (defining 2D polygon vertices in normalized or frame coordinates).
3. Create a verification helper to validate the polygon definition against video resolution.

### Verification Criteria
- [ ] Test video exists and its resolution/framerate is verified with `ffprobe` or equivalent.
- [ ] `configs/restricted_zone.json` exists with valid polygon vertex coordinates.

---

## Phase 2: PipeGen Requirement & Pipeline IR Specification

> **Objective**: Translate safety requirements into an official CraftifAI PipeGen Pipeline IR representation adhering strictly to PipeGen schemas.

### Steps
1. Author the safety monitoring pipeline requirements according to PipeGen's requirement specification format.
2. Generate the Pipeline IR specifying:
   - Source: Local video file input
   - Decoder: GPU hardware acceleration
   - Inference: Person detector model
   - Tracker: Object tracking component
   - Output/Sink: Frame metadata / detections
3. Save the specification to `artifacts/ir/pipeline_ir.yaml` (or `.json`, per PipeGen format).
4. Run PipeGen's IR validation tool to ensure schema compliance.

### Verification Criteria
- [ ] Pipeline IR file saved in `artifacts/ir/`.
- [ ] PipeGen IR validation passes without schema errors.

---

## Phase 3: Model Selection & TensorRT Compilation

> **Objective**: Compile the designated person detection model into an optimized TensorRT engine for the RTX 4060 and generate a benchmark report.

### Steps
1. Select the supported standard person detector model recommended by PipeGen / DeepStream.
2. Invoke the PipeGen model compiler to generate the TensorRT `.engine` file targeting compute capability 8.9 (RTX 4060).
3. Capture the full compilation log to `artifacts/benchmarks/engine_build.log`.
4. Run PipeGen benchmark tooling to measure:
   - Inference latency (ms)
   - Pipeline throughput (FPS)
   - GPU VRAM utilization
5. Save the performance results to `artifacts/benchmarks/speed_report.md`.

### Verification Criteria
- [ ] Valid TensorRT engine file compiled and verified.
- [ ] Build log and benchmark speed report produced in `artifacts/benchmarks/`.

---

## Phase 4: DeepStream Pipeline Execution via PipeGen

> **Objective**: Execute the end-to-end DeepStream perception pipeline using PipeGen's runtime on the test video.

### Steps
1. Launch the PipeGen execution command feeding the test video through the compiled DeepStream pipeline.
2. Verify GPU hardware acceleration is active (NVDEC + TensorRT inference on RTX 4060).
3. Verify that bounding boxes and track IDs are extracted per frame.

### Verification Criteria
- [ ] Pipeline executes to completion without crashes.
- [ ] Detection and tracking metadata stream is actively generated from the video frames.

---

## Phase 5: Zone Intrusion Detection & Actionable JSON Alerting

> **Objective**: Evaluate person tracks against the restricted zone polygon and generate actionable incident alerts.

### Steps
1. Implement the spatial intrusion logic:
   - Calculate person ground contact point (bottom-center of bounding box: `(x_center, y_max)`).
   - Test point-in-polygon containment against the configured restricted zone.
2. Structure the incident output as an actionable JSON schema:
   - `incident_id`: Unique identifier
   - `timestamp_iso`: Event timestamp
   - `frame_index`: Video frame number
   - `track_id`: Unique track identifier of the person
   - `zone_id`: Restricted zone identifier
   - `confidence`: Detection confidence score
   - `bbox`: Coordinates `[x_min, y_min, x_max, y_max]`
   - `severity`: Alert severity level (e.g., `WARNING`, `CRITICAL`)
3. Write incident logs to `data/output/incident_records.jsonl`.

### Verification Criteria
- [ ] Sample intrusion triggers an incident record when a person enters the polygon.
- [ ] Generated JSON records validate against the defined incident schema.
- [ ] No false alerts when a person stays outside the polygon.

---

## Phase 6: End-to-End Prototype Validation & Deliverable Packaging

> **Objective**: Conduct a comprehensive run, package all hackathon evidence, and finalize documentation.

### Steps
1. Execute the full end-to-end workflow: `Requirement -> Pipeline IR -> Model -> Compiler -> Run -> Alert Output`.
2. Inspect and verify all generated artifacts:
   - Pipeline IR file in `artifacts/ir/`
   - TensorRT engine build logs and benchmark report in `artifacts/benchmarks/`
   - Sample incident output in `data/output/`
3. Document exact reproduction steps in `README.md`.

### Verification Criteria
- [ ] Clean end-to-end execution reproducible with a single script or documented command.
- [ ] All mandatory hackathon deliverables present and verified in the repository.
