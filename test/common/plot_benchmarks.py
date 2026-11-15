"""Plot the checked-in benchmark results as plain bar charts."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "test" / "result"
ASSETS = ROOT / "assets" / "benchmarks"
QWEN_BACKENDS = {
    "transformers_tokens_per_second": "Transformers",
    "tileserve_tokens_per_second": "TileServe",
}


def read_result(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def save_plot(ax, filename, decimals, headroom=1.08):
    for bars in ax.containers:
        ax.bar_label(bars, fmt=f"%.{decimals}f", padding=3, fontsize=9)
    lower, upper = ax.get_ylim()
    ax.set_ylim(lower, upper * headroom)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(ASSETS / filename, dpi=150)
    plt.close(fig)


def qwen_throughput(report, index_key):
    rows = pd.DataFrame(report["results"]).set_index(index_key)
    return rows[list(QWEN_BACKENDS)].rename(columns=QWEN_BACKENDS)


def plot_qwen():
    prefill = qwen_throughput(read_result("qwen-prefill"), "prompt_tokens")
    prefill.index = pd.Index([f"{tokens:,}" for tokens in prefill.index], name="Prompt tokens")
    ax = prefill.plot.bar(rot=0, ylabel="Tokens/s", title="Qwen prefill")
    ax.legend(loc="upper left", ncol=2)
    save_plot(ax, "qwen-prefill.png", 0, headroom=1.20)

    report = read_result("qwen-decode-2048")
    decode = qwen_throughput(report, "batch_size").rename_axis("Batch size")
    ax = decode.plot.bar(
        figsize=(7, 4), width=0.75, rot=0, ylabel="Decode throughput (tokens/s)",
        title=f"Qwen decode at {report['prompt_tokens']:,} prompt tokens",
    )
    ax.legend(loc="upper left", ncol=2)
    save_plot(ax, "qwen-decode-batch.png", 1)


def plot_h3():
    times = pd.Series(read_result("h3-generation")["generation_seconds"])
    ax = times.plot.bar(rot=0, ylabel="Seconds", title="H3 generation time")
    save_plot(ax, "h3-generation.png", 2)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    plot_qwen()
    plot_h3()


if __name__ == "__main__":
    main()
