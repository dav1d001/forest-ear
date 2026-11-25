"""Turn results/containers_<N>_stats.csv files into a markdown table and a chart."""
import csv
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"


def read(n):
    with open(R / f"containers_{n}_stats.csv") as f:
        for row in csv.DictReader(f):
            if row["Name"] == "POST /api/predict":
                return row
    raise SystemExit(f"no predict row in containers_{n}_stats.csv")


def main():
    ns = sorted(int(p.stem.split("_")[1]) for p in R.glob("containers_*_stats.csv") if "history" not in p.stem)
    if not ns:
        sys.exit("No results found. Run scripts/run_load_tests.sh first.")
    rows = []
    for n in ns:
        r = read(n)
        rows.append((n, int(r["Request Count"]), int(r["Failure Count"]), float(r["Requests/s"]),
                     float(r["Average Response Time"]), float(r["50%"]), float(r["95%"]), float(r["99%"])))
    md = ["| Containers | Requests | Failures | Req/s | Avg (ms) | p50 (ms) | p95 (ms) | p99 (ms) |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {n} | {c} | {f} | {rps:.1f} | {avg:.0f} | {p50:.0f} | {p95:.0f} | {p99:.0f} |" for n, c, f, rps, avg, p50, p95, p99 in rows]
    (R / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        x = [r[0] for r in rows]
        ax[0].plot(x, [r[3] for r in rows], "o-", color="#1F6F66"); ax[0].set_title("Throughput (req/s)")
        ax[1].plot(x, [r[5] for r in rows], "o-", label="p50", color="#1F6F66"); ax[1].plot(x, [r[6] for r in rows], "s-", label="p95", color="#D4452B")
        ax[1].set_title("Latency (ms)"); ax[1].legend()
        for a in ax: a.set_xlabel("API containers"); a.set_xticks(x); a.grid(alpha=.25)
        plt.tight_layout(); plt.savefig(R / "load_test.png", dpi=130)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
