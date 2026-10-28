"""
Sentinel Vision AI - Reproducible Demo Launcher

Manages the execution of the workplace safety perception system:
1. Validates environment, configurations, and required deliverables.
2. Checks for compiled PipeGen DeepStream pipeline files in generated/.
3. Launches real pipeline execution or offline metadata stream consumption.
4. Exits cleanly with actionable error messages without simulating inference.
"""

import argparse
import glob
import os
import sys
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def check_deliverable(label: str, patterns: list) -> bool:
    if isinstance(patterns, str):
        patterns = [patterns]
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(os.path.join(PROJECT_ROOT, pat)))
    found = len(matches) > 0 and any(os.path.getsize(m) > 0 for m in matches if os.path.isfile(m))
    status_str = "[READY]" if found else "[PENDING]"
    details = f"({matches[0]})" if found else f"(patterns: {', '.join(patterns)})"
    print(f"  {status_str:<10} {label:<32} {details}")
    return found


def main():
    parser = argparse.ArgumentParser(description="Sentinel Vision AI Demo Launcher")
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "restricted_zone.json"), help="Zone config file")
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "data", "output", "incident_records.jsonl"), help="Incident output log")
    parser.add_argument("--stream", default=None, help="Input detection stream JSONL file (optional)")
    parser.add_argument("--pipeline-dir", default=os.path.join(PROJECT_ROOT, "generated"), help="Directory containing PipeGen generated pipeline")
    parser.add_argument("--replay-pipegen", action="store_true", help="Replay genuine PipeGen metadata.json through Sentinel Safety Adapter")
    parser.add_argument("--render-video", action="store_true", help="Render video overlay demo from authentic PipeGen detections")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("        SENTINEL VISION AI — WORKPLACE SAFETY MONITORING SYSTEM")
    print("        Track A: CraftifAI PipeGen Perception Submission")
    print("=" * 80)

    # 1. Verification of configurations
    print("\n[PRE-FLIGHT] Checking Configurations & Schemas:")
    if not os.path.isfile(args.config):
        print(f"  [ERROR] Restricted zone configuration missing: {args.config}")
        sys.exit(1)
    print(f"  [OK] Restricted Zone Config: {args.config}")

    schema_file = os.path.join(PROJECT_ROOT, "configs", "incident_schema.json")
    if not os.path.isfile(schema_file):
        print(f"  [ERROR] Incident JSON Schema missing: {schema_file}")
        sys.exit(1)
    print(f"  [OK] Incident JSON Schema:   {schema_file}")

    # 2. Output directory verification
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"  [OK] Incident Records Sink:  {args.output}")

    # 3. Deliverables Status Check
    print("\n[INVENTORY] Hackathon Deliverables Status:")
    ir_ok = check_deliverable("PipeGen Pipeline IR", ["artifacts/ir/*ir*.*", "artifacts/pipegen-export/*IR*.json"])
    eng_ok = check_deliverable("TensorRT Engine (.plan / .engine)", ["artifacts/benchmarks/*.plan", "artifacts/benchmarks/*.engine", "artifacts/engines/*.plan"])
    bench_ok = check_deliverable("Benchmark Speed Report", ["artifacts/benchmarks/speed_report.md"])
    pipe_ok = check_deliverable("Generated Pipeline Code", ["generated/*", "artifacts/pipegen-export/pipeline/*"])

    print("\n" + "-" * 80)

    # 4. Execution Dispatch
    pipegen_meta = os.path.join(PROJECT_ROOT, "artifacts", "pipegen-export", "output", "metadata.json")

    # Mode 1: Render Video Overlay Demo
    if args.render_video:
        print("[ACTIVE] Video Render Mode: Generating verified demonstration overlay MP4 (sequential frames)...\n")
        sys.stdout.flush()
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts", "render_repaired_demo.py"),
            "--config", args.config,
            "--metadata", pipegen_meta,
        ]
        subprocess.run(cmd)

    # Mode 2: Explicit Stream Ingestion Mode
    elif args.stream:
        if args.stream != "-" and not os.path.isfile(args.stream):
            print(f"[ERROR] Specified stream file does not exist: {args.stream}")
            sys.exit(1)
        print(f"[ACTIVE] Ingestion Mode: Processing detection stream from '{args.stream}'...\n")
        sys.stdout.flush()
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "src", "sentinel_pipeline.py"),
            "--config", args.config,
            "--output", args.output,
            "--stream", args.stream,
        ]
        subprocess.run(cmd)

    # Mode 3: Replay Genuine PipeGen Metadata
    elif args.replay_pipegen or os.path.isfile(pipegen_meta):
        print("[ACTIVE] PipeGen Perception Bridge Mode: Ingesting genuine DeepStream detections...\n")
        sys.stdout.flush()
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts", "pipegen_to_sentinel.py"),
            "--config", args.config,
            "--output", args.output,
            "--metadata", pipegen_meta,
        ]
        subprocess.run(cmd)

    # Mode 4: Standby
    else:
        print("[STANDBY] Sentinel Vision AI is in verification standby.")
        print("\n[ACTIONABLE NEXT STEPS]:")
        print("  1. Replay authentic PipeGen detections: python launch_demo.py --replay-pipegen")
        print("  2. Render visual demonstration video:   python launch_demo.py --render-video")
        print("  3. Run all automated integration tests: python -m pytest tests/ -v")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
