import argparse
import time

from src import (baseline, config, data_download, drift_analysis, eda, experiments, explain, imbalance_ablation,
                 make_dataset, policy_comparison, stats_tests)

STEPS = ["download", "clean", "eda", "baseline", "imbalance", "drift", "policies", "explain", "extras", "stats", "readme"]
DEFAULT_STEPS = [s for s in STEPS if s not in ("policies", "explain", "extras", "stats")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", default=DEFAULT_STEPS, choices=STEPS)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--policies", nargs="+", default=["no_retrain", "fixed:4"])
    a = ap.parse_args()
    config.ensure_dirs()
    steps = [s for s in a.steps if not (s == "download" and a.skip_download)]

    def run(name, fn):
        if name in steps:
            t = time.time()
            print(f"\n===== {name} =====")
            fn()
            print(f"===== {name} done in {time.time() - t:.0f}s =====")

    run("download", data_download.main)
    run("clean", make_dataset.build)
    run("eda", eda.main)
    run("baseline", lambda: baseline.main(tuple(a.policies)))
    run("imbalance", imbalance_ablation.main)
    run("drift", drift_analysis.main)
    run("policies", policy_comparison.main)
    run("explain", explain.main)
    run("extras", experiments.main)
    run("stats", stats_tests.main)
    if "readme" in steps:
        import sys
        from src import make_readme
        sys.argv = [sys.argv[0]]
        make_readme.main()


if __name__ == "__main__":
    main()
