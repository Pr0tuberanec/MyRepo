#!/usr/bin/env python3
"""Plot PTDE vs PTDE++ from demo_ptde_compare.json (marl-eval aggregate scores)."""

from pathlib import Path

from matplotlib import pyplot as plt

from plotting_import import get_load_and_merge_json_dicts, get_plotting

Plotting = get_plotting()
load_and_merge_json_dicts = get_load_and_merge_json_dicts()

JSON_PATH = Path(__file__).resolve().parent / "demo_ptde_compare.json"
OUT_DIR = Path(__file__).resolve().parent / "output"
LEGEND = {"mappo": "PTDE++", "ippo": "PTDE"}


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    raw = load_and_merge_json_dicts([str(JSON_PATH)])
    processed = Plotting.process_data(raw, metrics_to_normalize=["return"])
    env_cm, se_cm = Plotting.create_matrices(
        processed, env_name="camar", metrics_to_normalize=["return"]
    )

    fig_agg, scores, cis = Plotting.aggregate_scores(
        env_cm,
        metric_name="return",
        metrics_to_normalize=["return"],
        legend_map=LEGEND,
        tabular_results_file_path=str(OUT_DIR / "aggregated_score"),
        save_tabular_as_latex=False,
    )
    fig_agg.savefig(OUT_DIR / "aggregate_scores_return.png", dpi=150, bbox_inches="tight")
    print("Saved:", OUT_DIR / "aggregate_scores_return.png")
    print("\nAggregate scores:")
    for algo, metrics in scores.items():
        print(f"  {algo}:")
        for m, v in metrics.items():
            lo, hi = cis[algo][m]
            print(f"    {m}: {v:.3f}  CI [{lo:.3f}, {hi:.3f}]")

    fig_pp = Plotting.performance_profile_figure(
        env_cm,
        metric_name="return",
        metrics_to_normalize=["return"],
        legend_map=LEGEND,
    )
    fig_pp.savefig(OUT_DIR / "performance_profile_return.png", dpi=150, bbox_inches="tight")
    print("Saved:", OUT_DIR / "performance_profile_return.png")

    fig_se, _, _ = Plotting.environemnt_sample_efficiency_curves(
        se_cm,
        metric_name="return",
        metrics_to_normalize=["return"],
        legend_map=LEGEND,
    )
    fig_se.savefig(OUT_DIR / "sample_efficiency_return.png", dpi=150, bbox_inches="tight")
    print("Saved:", OUT_DIR / "sample_efficiency_return.png")

    plt.close("all")


if __name__ == "__main__":
    main()
