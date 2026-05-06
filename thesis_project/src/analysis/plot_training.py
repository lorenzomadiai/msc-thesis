import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


METRICS = [
    ("AverageEpRet", "Average Episode Return", "compare_averageepret.png"),
    ("AverageEpCost", "Average Episode Cost", "compare_averageepcost.png"),
    ("CostRate", "Cost Rate", "compare_costrate.png"),
    ("TestEpGoals", "Test Episode Goals", "compare_testepgoals.png"),
    ("TestEpLen", "Test Episode Length", "compare_testeplen.png"),
]


def smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()


def read_log(path: str) -> pd.DataFrame:
    # I tuoi file sono tab-separated: sep="\t"
    df = pd.read_csv(path, sep="\t")
    if "Epoch" not in df.columns:
        raise KeyError(f"Column 'Epoch' not found in {path}. Columns: {list(df.columns)}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Compare 3 trainings from progress logs (TSV).")
    parser.add_argument("--log1", required=True, help="Path to first log (e.g., run1/progress.csv)")
    parser.add_argument("--log2", required=True, help="Path to second log")
    parser.add_argument("--log3", required=True, help="Path to third log")
    parser.add_argument("--label1", default="run1", help="Legend label for log1")
    parser.add_argument("--label2", default="run2", help="Legend label for log2")
    parser.add_argument("--label3", default="run3", help="Legend label for log3")
    parser.add_argument("--smooth-window", type=int, default=1, help="Moving average window (1 = off)")
    parser.add_argument("--plots-dir", default="plots", help="Output folder for plots")
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    logs = [
        (args.log1, args.label1, read_log(args.log1)),
        (args.log2, args.label2, read_log(args.log2)),
        (args.log3, args.label3, read_log(args.log3)),
    ]

    # Check that required metric columns exist
    for metric, _, _ in METRICS:
        missing = [p for (p, _, df) in logs if metric not in df.columns]
        if missing:
            raise KeyError(f"Missing column '{metric}' in: {missing}")

    for metric, ylabel, out_name in METRICS:
        plt.figure()

        for (path, label, df) in logs:
            x = df["Epoch"].to_numpy()
            y = smooth(df[metric], args.smooth_window).to_numpy()
            plt.plot(x, y, label=label)

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        title = f"Epoch vs {metric}"
        if args.smooth_window > 1:
            title += f" (window={args.smooth_window})"
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.plots_dir, out_name))
        plt.close()


if __name__ == "__main__":
    main()
