# Traces the appropriate pinned Mathlib revision with LeanDojo-v2 and apply the local runtime 
# patches this repo expects. The resulting benchmark directory is consumed later by `process_herald.py`.

from pathlib import Path
import filecmp
import os
import shutil
import site
import sys

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATHLIB_URL = "https://github.com/leanprover-community/mathlib4"
MATHLIB_COMMIT = "20c73142afa995ac9c8fb80a9bb585a55ca38308"
BENCHMARK_DIR = PROJECT_ROOT / "raid" / "data" / f"mathlib4_{MATHLIB_COMMIT}"


# List likely site-packages directories for the active Python environment.
def _candidate_site_packages() -> list[Path]:
    candidates = [Path(p) for p in site.getsitepackages()]
    candidates.append(Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")
    return candidates


# Find an installed LeanDojo-v2 file by its path relative to site-packages.
# relative_path: Package-relative file path to locate.
def _find_package_file(relative_path: str) -> Path:
    for base in _candidate_site_packages():
        candidate = base / relative_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find installed LeanDojo-v2 file {relative_path}")


# Apply the local LeanDojo-v2 file patches required by this repository.
def patch_leandojo_runtime() -> None:
    extractor_src = PROJECT_ROOT / "scripts" / "ExtractData.lean"
    extractor_dst = _find_package_file("lean_dojo_v2/lean_dojo/data_extraction/ExtractData.lean")
    if not filecmp.cmp(extractor_src, extractor_dst, shallow=False):
        shutil.copyfile(extractor_src, extractor_dst)
        print(f"Patched LeanDojo ExtractData.lean at {extractor_dst}")
    else:
        print(f"LeanDojo ExtractData.lean already matches {extractor_src}")

    constants = _find_package_file("lean_dojo_v2/utils/constants.py")
    text = constants.read_text(encoding="utf-8")
    old = 'os.environ["RAY_TMPDIR"] = f"/tmp/ray"'
    new = 'os.environ["RAY_TMPDIR"] = os.environ.get("RAY_TMPDIR", "/tmp/ray")'
    if old in text:
        constants.write_text(text.replace(old, new), encoding="utf-8")
        print(f"Patched LeanDojo RAY_TMPDIR handling at {constants}")
    else:
        print(f"LeanDojo RAY_TMPDIR handling already patched at {constants}")


# Find the most recently modified temporary Mathlib trace work directory.
def latest_trace_workdir() -> Path | None:
    tmp_dir = Path(os.environ.get("TMP_DIR", "/tmp"))
    candidates = sorted(
        (p / "mathlib4" for p in tmp_dir.glob("tmp_*/") if (p / "mathlib4").exists()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


# Check whether the expected LeanDojo benchmark directory already exists and is usable.
def existing_benchmark_is_usable() -> bool:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from validate_leandojo_benchmark import validate_benchmark

    if not BENCHMARK_DIR.exists():
        return False
    report = validate_benchmark(BENCHMARK_DIR)
    if report["ok"]:
        print(f"LeanDojo benchmark already exists at {BENCHMARK_DIR}")
        print("Skipping trace; process_herald.py can consume this directory directly.")
        print("Benchmark counts:")
        print(report["counts"])
        return True
    print(f"Found incomplete LeanDojo benchmark at {BENCHMARK_DIR}:")
    print(report)
    return False

# Prepare the environment, trace the pinned Mathlib revision, and report the result.
def main():
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    if "TMP_DIR" in os.environ:
        Path(os.environ["TMP_DIR"]).mkdir(parents=True, exist_ok=True)
    if "RAY_TMPDIR" in os.environ:
        Path(os.environ["RAY_TMPDIR"]).mkdir(parents=True, exist_ok=True)

    if existing_benchmark_is_usable():
        return

    patch_leandojo_runtime()

    from lean_dojo_v2.database import DynamicDatabase
    
    print("Initializing LeanDojo-v2 Dynamic Database...")
    database = DynamicDatabase()
    
    print(f"Tracing Mathlib4 at commit {MATHLIB_COMMIT}...")
    print("Estimated time 10-60 minutes...")
    
    try:
        database.trace_repository(
            url=MATHLIB_URL,
            commit=MATHLIB_COMMIT,
            build_deps=False,
        )
    except AssertionError:
        workdir = latest_trace_workdir()
        print("\nLeanDojo-v2 failed its post-extraction file-count assertion.")
        print("This usually means build/ir contains extra/stale .ast.json or .dep_paths files")
        print("that do not correspond to .olean files in build/lib/lean.")
        if workdir is not None:
            print(f"Latest trace workdir appears to be: {workdir}")
            print("Run:")
            print(f"  python scripts/diagnose_leandojo_trace.py --repo {workdir}")
        raise
    
    print("\nTrace complete! Mathlib is safely indexed in the raid/ database.")

if __name__ == "__main__":
    main()
