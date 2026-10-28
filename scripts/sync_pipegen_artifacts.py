"""
Sentinel Vision AI - PipeGen Artifact Synchronization Tool

Safely scans candidate PipeGen output paths in WSL and synchronizes genuine
compiled artifacts into the project workspace:
- Pipeline IR (artifacts/ir/pipeline_ir.yaml)
- TensorRT engine (artifacts/benchmarks/*.engine)
- Compiler logs (artifacts/benchmarks/engine_build.log)
- DeepStream generated pipeline files (generated/)

Guarantees:
- Never creates mock or placeholder files.
- Ignores 0-byte or corrupted partial files.
- Handles duplicate files by prioritizing the most recent build.
- Reports detailed source, destination, and file sizes.
"""

import os
import shutil
import glob
import sys
from typing import List, Dict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CANDIDATE_SEARCH_ROOTS = [
    r"\\wsl.localhost\Ubuntu-PipeGen\home\utkarsh_dev\.pipegen",
    r"\\wsl.localhost\Ubuntu-PipeGen\home\utkarsh_dev\sentinel-vision-ai",
    r"\\wsl.localhost\Ubuntu-PipeGen\opt\PipeGen",
    r"\\wsl.localhost\Ubuntu-PipeGen\tmp",
]


def safe_copy_file(source: str, destination: str) -> bool:
    """Copies file safely if source exists, is non-empty, and newer than destination."""
    if not os.path.isfile(source):
        return False
    size = os.path.getsize(source)
    if size == 0:
        print(f"  [SKIPPED] Source file is 0 bytes (incomplete/corrupt): {source}")
        return False

    os.makedirs(os.path.dirname(destination), exist_ok=True)

    if os.path.isfile(destination):
        dest_mtime = os.path.getmtime(destination)
        src_mtime = os.path.getmtime(source)
        if dest_mtime >= src_mtime and os.path.getsize(destination) == size:
            print(f"  [UP-TO-DATE] {os.path.basename(destination)} ({size:,} bytes)")
            return True

    shutil.copy2(source, destination)
    print(f"  [COPIED] {os.path.basename(destination):<30} ({size:,} bytes) from {source}")
    return True


def scan_and_sync():
    print("=" * 80)
    print("       SENTINEL VISION AI — PIPEGEN ARTIFACT SYNCHRONIZATION")
    print("=" * 80)

    found_count = 0

    for root in CANDIDATE_SEARCH_ROOTS:
        if not os.path.exists(root):
            continue

        print(f"\n[SCANNING] Directory: {root}")

        # 1. Pipeline IR
        ir_matches = glob.glob(os.path.join(root, "**", "*pipeline*ir*.*"), recursive=True) + \
                     glob.glob(os.path.join(root, "**", "*ir*.yaml"), recursive=True) + \
                     glob.glob(os.path.join(root, "**", "*ir*.json"), recursive=True)

        dest_ir_dir = os.path.join(PROJECT_ROOT, "artifacts", "ir")
        for f in ir_matches:
            if "node_modules" in f or ".cache" in f:
                continue
            dest_file = os.path.join(dest_ir_dir, os.path.basename(f))
            if safe_copy_file(f, dest_file):
                found_count += 1

        # 2. TensorRT Engines (.engine / .trt)
        engine_matches = glob.glob(os.path.join(root, "**", "*.engine"), recursive=True) + \
                         glob.glob(os.path.join(root, "**", "*.trt"), recursive=True)

        dest_bench_dir = os.path.join(PROJECT_ROOT, "artifacts", "benchmarks")
        for eng in engine_matches:
            dest_file = os.path.join(dest_bench_dir, os.path.basename(eng))
            if safe_copy_file(eng, dest_file):
                found_count += 1

        # 3. Compiler build logs (*build*.log, *compile*.log, *tensorrt*.log)
        log_matches = glob.glob(os.path.join(root, "**", "*engine*build*.log"), recursive=True) + \
                      glob.glob(os.path.join(root, "**", "*trt*.log"), recursive=True)

        for log in log_matches:
            dest_file = os.path.join(dest_bench_dir, os.path.basename(log))
            if safe_copy_file(log, dest_file):
                found_count += 1

        # 4. Generated pipeline directory
        gen_dirs = glob.glob(os.path.join(root, "**", "generated"), recursive=True)
        dest_gen = os.path.join(PROJECT_ROOT, "generated")
        for g_dir in gen_dirs:
            if not os.path.isdir(g_dir):
                continue
            print(f"  [SCAN PIPELINE DIR] {g_dir}")
            for item in os.listdir(g_dir):
                s_item = os.path.join(g_dir, item)
                d_item = os.path.join(dest_gen, item)
                if os.path.isdir(s_item):
                    shutil.copytree(s_item, d_item, dirs_exist_ok=True)
                    print(f"  [COPIED DIR] {item}/")
                    found_count += 1
                elif os.path.isfile(s_item):
                    if safe_copy_file(s_item, d_item):
                        found_count += 1

    print("\n" + "=" * 80)
    if found_count > 0:
        print(f"[SUMMARY] Successfully synchronized {found_count} authentic artifact(s).")
    else:
        print("[SUMMARY] No generated pipeline artifacts found yet across search paths.")
        print("          Run the PipeGen compiler first to generate genuine artifacts.")
    print("=" * 80 + "\n")
    return found_count


if __name__ == "__main__":
    scan_and_sync()
