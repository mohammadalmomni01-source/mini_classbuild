"""
scripts/run_evaluation.py
Runs every test listed in SCENARIOS once, offline, and reports each scenario
as passed or failed, with an optional JSON report in evaluation_results/.
"""
import argparse
import json
import os
import socket
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation_scenarios import SCENARIOS, Scenario

DEFAULT_OUTPUT_DIR = "evaluation_results"


# ---------------------------------------------------------------------------
# Network isolation
# ---------------------------------------------------------------------------

def block_network():
    """Patches socket so any connection outside this computer raises OSError."""
    orig_connect = socket.socket.connect

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        # Allow loopback connections only
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise OSError(f"Network access is blocked during offline evaluation: {address}")
        return orig_connect(self, address)

    socket.socket.connect = guarded_connect


# ---------------------------------------------------------------------------
# Custom test result recorder
# ---------------------------------------------------------------------------

class RecordingResult(unittest.TestResult):
    """
    Records outcome and failure reason for each test id.
    Outcome: 'passed', 'failed', 'error', 'skipped'
    """
    def __init__(self):
        super().__init__()
        self.outcomes: dict[str, str] = {}
        self.reasons: dict[str, str] = {}

    def _test_id(self, test):
        # Full id: tests.test_module.TestCase.method
        return test.id()

    def addSuccess(self, test):
        super().addSuccess(test)
        self.outcomes[self._test_id(test)] = "passed"

    def addFailure(self, test, err):
        super().addFailure(test, err)
        tid = self._test_id(test)
        self.outcomes[tid] = "failed"
        self.reasons[tid] = str(err[1])

    def addError(self, test, err):
        super().addError(test, err)
        tid = self._test_id(test)
        self.outcomes[tid] = "error"
        self.reasons[tid] = str(err[1])

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        tid = self._test_id(test)
        self.outcomes[tid] = "skipped"
        self.reasons[tid] = reason


# ---------------------------------------------------------------------------
# Test loading & execution
# ---------------------------------------------------------------------------

def load_tests(test_ids: list[str]) -> tuple[unittest.TestSuite, dict[str, str]]:
    """
    Load tests by their dotted IDs.
    Returns (suite, load_errors).
    An ID that doesn't exist is recorded as a load error, never ignored.
    """
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    load_errors: dict[str, str] = {}

    for tid in test_ids:
        try:
            loaded = loader.loadTestsFromName(tid)
            # Check if loader created an error test case
            cases = list(loaded)
            if cases and isinstance(cases[0], unittest.loader._FailedTest):
                exc = getattr(cases[0], "_exception", None)
                load_errors[tid] = str(exc) if exc else "Failed to load test"
            else:
                suite.addTests(loaded)
        except Exception as exc:
            load_errors[tid] = str(exc)

    return suite, load_errors


def short(test_id: str) -> str:
    """Shorten test ID: tests.test_foo.Bar.test_baz -> Bar.test_baz"""
    parts = test_id.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else test_id


def evaluate() -> dict:
    """Runs all unique tests once and builds the evaluation report."""
    block_network()

    # Collect all unique test IDs
    all_test_ids: list[str] = []
    for s in SCENARIOS:
        for tid in s.tests:
            if tid not in all_test_ids:
                all_test_ids.append(tid)

    suite, load_errors = load_tests(all_test_ids)
    recorder = RecordingResult()
    suite.run(recorder)

    # Combine recorded outcomes and load errors
    outcomes = dict(recorder.outcomes)
    reasons = dict(recorder.reasons)
    for tid, err in load_errors.items():
        outcomes[tid] = "error"
        reasons[tid] = err

    scenario_reports = []
    total_passed = 0

    for s in SCENARIOS:
        tests_passed = []
        failures = []
        for tid in s.tests:
            status = outcomes.get(tid, "missing")
            if status == "passed":
                tests_passed.append(tid)
            else:
                reason = reasons.get(tid, f"Status: {status}")
                failures.append(f"{short(tid)}: {reason}")

        # Scenario passes ONLY IF all listed tests ran and passed
        passed = (len(tests_passed) == len(s.tests) and len(s.tests) > 0)
        if passed:
            total_passed += 1

        scenario_reports.append({
            "number": s.number,
            "name": s.name,
            "category": s.category,
            "checks": s.checks,
            "tests": s.tests,
            "tests_passed": tests_passed,
            "passed": passed,
            "failures": failures,
        })

    total_scenarios = len(SCENARIOS)
    percentage = (total_passed / total_scenarios * 100.0) if total_scenarios else 0.0

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_scenarios": total_scenarios,
        "passed_scenarios": total_passed,
        "failed_scenarios": total_scenarios - total_passed,
        "pass_percentage": round(percentage, 1),
        "scenarios": scenario_reports,
    }


def print_report(report: dict):
    print("=" * 80)
    print(" Mini ClassBuild — Offline Evaluation Report")
    print("=" * 80)

    # Group by category
    categories: dict[str, list] = {}
    for s in report["scenarios"]:
        categories.setdefault(s["category"], []).append(s)

    for cat, items in categories.items():
        print(f"\n[{cat}]")
        for item in items:
            mark = "PASS" if item["passed"] else "FAIL"
            print(f"  [{mark}] #{item['number']:02d}: {item['name']}")
            print(f"         {item['checks']}")
            if not item["passed"]:
                for f in item["failures"]:
                    print(f"         -> Error: {f}")

    print("\n" + "-" * 80)
    print(f"Total scenarios:  {report['total_scenarios']}")
    print(f"Passed:           {report['passed_scenarios']}")
    print(f"Failed:           {report['failed_scenarios']}")
    print(f"Pass rate:        {report['pass_percentage']:.1f}%")
    print("=" * 80)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Mini ClassBuild evaluation scenarios.")
    parser.add_argument("--no-json", action="store_true", help="Do not save JSON reports to disk.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save JSON reports.")
    args = parser.parse_args(argv)

    report = evaluate()
    print_report(report)

    if not args.no_json:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        time_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_file = out_dir / f"evaluation-{time_str}.json"
        latest_file = out_dir / "latest.json"

        content = json.dumps(report, indent=2)
        run_file.write_text(content, encoding="utf-8")
        latest_file.write_text(content, encoding="utf-8")
        print(f"\nSaved report to {run_file} and {latest_file}")

    sys.exit(0 if report["failed_scenarios"] == 0 else 1)


if __name__ == "__main__":
    main()
