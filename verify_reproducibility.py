"""Code verification: re-run the WHOLE pipeline in a temporary folder and check that
every headline number matches the results committed in /results.

Usage:  python verify_reproducibility.py
Exit code 0 = reproduced, 1 = mismatch.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOL = 1e-3   # absolute tolerance on rounded metrics (allows tiny cross-platform float noise)


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list):
            out[key] = v
        else:
            out[key] = v
    return out


def close(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= max(TOL, 1e-3 * abs(b))
    return a == b


def main():
    files = ["dataset_summary.json", "model_summary.json", "optimization_summary.json"]
    skip = {"runtime_seconds"}
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, LOGISTICS_OUTPUT_DIR=tmp)
        print(f"Re-running pipeline into temporary folder: {tmp}")
        subprocess.run([sys.executable, str(ROOT / "src" / "run_pipeline.py")], env=env, check=True,
                       stdout=subprocess.DEVNULL)
        bad, checked = [], 0
        for fn in files:
            ref = flatten(json.load(open(ROOT / "results" / fn)))
            new = flatten(json.load(open(Path(tmp) / "results" / fn)))
            for k, v in ref.items():
                if k.split(".")[-1] in skip:
                    continue
                checked += 1
                if k not in new or not close(new[k], v):
                    bad.append((fn, k, v, new.get(k)))
        # dataset itself must be byte-identical
        same_csv = (ROOT / "data" / "delivery_data.csv").read_bytes() == (Path(tmp) / "data" / "delivery_data.csv").read_bytes()
    print(f"Dataset CSV identical: {same_csv}")
    print(f"Metrics compared: {checked}, mismatches: {len(bad)}")
    for b in bad[:10]:
        print("  MISMATCH", b)
    ok = same_csv and not bad
    if ok:
        import datetime, platform
        import numpy, pandas, sklearn
        msg = (f"PASS. The full pipeline was re-run from scratch on {datetime.date.today().isoformat()} "
               f"(Python {platform.python_version()}, numpy {numpy.__version__}, pandas {pandas.__version__}, "
               f"scikit-learn {sklearn.__version__}). The regenerated dataset was byte-identical and all "
               f"{checked} compared result values matched the committed results (mismatches: 0).")
        (ROOT / "results" / "verification_status.txt").write_text(msg + "\n")
    print("RESULT:", "PASS - results reproduce exactly" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
