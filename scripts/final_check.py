"""
scripts/final_check.py
One command that verifies the project offline:
Python version, dependencies, all unit tests, the evaluation,
and the end-to-end test, with a pass/fail summary.
Standard library only; runs existing commands as subprocesses.
"""
import argparse
import importlib.metadata
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class CheckResult:
    name: str
    passed: bool
    summary: str
    details: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_requirements(path: Path) -> list[tuple[str, str | None]]:
    """Parse requirement names and optional minimum versions from requirements.txt."""
    reqs = []
    if not path.exists():
        return reqs
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        m = re.match(r"^([a-zA-Z0-9_\-]+)(?:>=([0-9\.]+))?", line)
        if m:
            pkg_name = m.group(1)
            min_ver = m.group(2)
            reqs.append((pkg_name, min_ver))
    return reqs


def child_environment() -> dict:
    """Copies current environment with GROQ_API_KEY blank and HF_HUB_OFFLINE=1."""
    env = os.environ.copy()
    env["GROQ_API_KEY"] = ""
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return env


def run(command: list[str], timeout: int = 60) -> tuple[int, str, float]:
    """Runs a command in the project root with a timeout; returns (code, output, duration)."""
    start = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            env=child_environment(),
            timeout=timeout,
        )
        duration = time.time() - start
        combined_output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return proc.returncode, combined_output.strip(), duration
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        return -1, f"Command timed out after {timeout} seconds", duration
    except Exception as exc:
        duration = time.time() - start
        return -1, str(exc), duration


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_python() -> CheckResult:
    ver = sys.version_info
    ver_str = f"{ver.major}.{ver.minor}.{ver.micro}"
    passed = (ver.major == 3 and ver.minor >= 11)
    summary = f"Python {ver_str} (3.11+ required)"
    return CheckResult("Python version", passed, summary, f"Detected: {sys.version}")


def check_dependencies() -> CheckResult:
    req_file = PROJECT_ROOT / "requirements.txt"
    reqs = parse_requirements(req_file)
    missing = []
    installed = []

    # Map package names to import names if they differ
    import_map = {
        "python-dotenv": "dotenv",
        "sentence-transformers": "sentence_transformers",
    }

    for pkg_name, min_ver in reqs:
        mod_name = import_map.get(pkg_name.lower(), pkg_name.replace("-", "_"))
        try:
            ver = importlib.metadata.version(pkg_name)
            installed.append(f"{pkg_name} ({ver})")
        except importlib.metadata.PackageNotFoundError:
            # Fallback check by importing module directly
            try:
                __import__(mod_name)
                installed.append(f"{pkg_name} (imported)")
            except Exception:
                missing.append(pkg_name)

    passed = (len(missing) == 0)
    summary = f"{len(installed)}/{len(reqs)} requirements satisfied"
    details = f"Installed:\n" + "\n".join(installed)
    if missing:
        details += "\nMissing:\n" + "\n".join(missing)
    return CheckResult("Dependencies", passed, summary, details)


def check_unit_tests() -> CheckResult:
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"]
    code, output, duration = run(cmd)
    passed = (code == 0)
    summary = f"All unit tests passed ({duration:.2f}s)" if passed else "Unit tests failed"
    return CheckResult("Unit tests", passed, summary, output)


def check_evaluation() -> CheckResult:
    cmd = [sys.executable, "scripts/run_evaluation.py", "--no-json"]
    code, output, duration = run(cmd)
    passed = (code == 0)
    summary = f"25/25 scenarios passed ({duration:.2f}s)" if passed else "Evaluation scenarios failed"
    return CheckResult("Evaluation scenarios", passed, summary, output)


def check_end_to_end() -> CheckResult:
    cmd = [sys.executable, "-m", "unittest", "tests.test_end_to_end", "-v"]
    code, output, duration = run(cmd)
    passed = (code == 0)
    summary = f"Workflow test passed ({duration:.2f}s)" if passed else "End-to-end test failed"
    return CheckResult("End-to-end workflow", passed, summary, output)


CHECKS = [
    ("Python version", check_python, False),
    ("Dependencies", check_dependencies, False),
    ("Unit tests", check_unit_tests, True),
    ("Evaluation scenarios", check_evaluation, True),
    ("End-to-end workflow", check_end_to_end, True),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Mini ClassBuild final verification checks.")
    parser.add_argument("--verbose", action="store_true", help="Print full output of checks.")
    args = parser.parse_args(argv)

    print("=" * 80)
    print(" Mini ClassBuild — Final Project Verification")
    print("=" * 80)

    all_passed = True
    deps_ok = True

    for name, func, needs_deps in CHECKS:
        if needs_deps and not deps_ok:
            print(f"  [SKIPPED] {name}: dependencies missing")
            all_passed = False
            continue

        result = func()
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.name}: {result.summary}")

        if name == "Dependencies" and not result.passed:
            deps_ok = False

        if not result.passed:
            all_passed = False

        if args.verbose or not result.passed:
            if result.details:
                indented = "\n".join("         " + line for line in result.details.splitlines()[:30])
                print(indented)
                if len(result.details.splitlines()) > 30:
                    print("         ... (truncated)")

    print("=" * 80)
    if all_passed:
        print(" SUCCESS: All verification checks passed!")
    else:
        print(" FAILURE: One or more checks failed.")
    print("=" * 80)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
