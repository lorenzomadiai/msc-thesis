#!/usr/bin/env python3
"""
cvar_exp1and2.py

Per ogni agente presente nei CSV di input calcola:
  - n_episodes  : numero totale di episodi
  - mean_cost   : media dei costi totali
  - cvar_cost   : CVaR_alpha = media dei ceil(N*alpha) episodi con costo peggiore
  - success_rate: frazione di episodi con goal raggiunto

Non viene fatta alcuna suddivisione per budget: tutti gli episodi dell'agente
sono usati insieme.

Se vengono passati più CSV (--csvs) vengono trattati come run indipendenti e
i risultati vengono aggregati con media ± SE (o ± std con --band std).

Esempio:
  python src/data_collection/EXP1&2/cvar_exp1and2.py \\
      --csvs results/EXP1/traindist_timeaware_seed2208_eps300_Bmin140_Bmax260_H260_5agents_my_run.csv \\
      --alpha 0.1 \\
      --out_csv results/EXP1/cvar_exp1.csv \\
      --out_dir plots/EXP1/cvar

  # Solo episodi completati anche dal conservative agent:
  python src/data_collection/EXP1&2/cvar_exp1and2.py \\
      --csvs results/EXP1/5seeds/all_runs_per_seed/*.csv \\
      --alpha 0.1 \\
      --conservative_filter \\
      --out_csv results/EXP1/cvar_conserv_filtered.csv \\
      --out_dir plots/EXP1/cvar_conserv_filtered
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Agent sort helper
# ---------------------------------------------------------------------------

def _penalty_sort_key(name: str):
    """
    Sort agents by penalty value extracted from names like p10, p30, p10_s29.
    Non-numeric names (aggressive, conservative) come first, alphabetically.
    """
    m = re.search(r"p(\d+)", str(name))
    if m:
        return (1, int(m.group(1)), str(name))
    return (0, 0, str(name))


def _get_penalty(name: str):
    m = re.search(r"p(\d+)", str(name))
    return int(m.group(1)) if m else None


# Decade buckets: (label, low_inclusive, high_inclusive)
DECADES = [
    ("p10\u2013p100",   10,    99),
    ("p100\u2013p1k",   100,   999),
    ("p1k\u2013p10k",   1000,  10000),
]


def build_decade_agg(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses the aggregated per-agent DataFrame into decade buckets.
    For each metric column ending in _mean/_band, computes:
      - decade_mean = mean of agent means within the decade
      - decade_band = SEM across agents within the decade
    Baseline agents (no penalty number) are kept as-is.
    """
    agents     = agg_df["agent"].tolist()
    baselines  = [a for a in agents if _get_penalty(a) is None]
    pen_agents = [a for a in agents if _get_penalty(a) is not None]

    mean_cols = [c for c in agg_df.columns if c.endswith("_mean")]
    band_cols = [c for c in agg_df.columns if c.endswith("_band")]
    extra_cols = [c for c in agg_df.columns
                  if c != "agent" and c not in mean_cols and c not in band_cols]

    rows = []

    # baselines unchanged
    for a in baselines:
        row_data = agg_df[agg_df["agent"] == a].iloc[0].to_dict()
        rows.append(row_data)

    # aggregate per decade
    for label, lo, hi in DECADES:
        in_decade = [a for a in pen_agents if lo <= _get_penalty(a) <= hi]
        if not in_decade:
            continue
        sub = agg_df[agg_df["agent"].isin(in_decade)]
        row = {"agent": label}
        for mc in mean_cols:
            vals = pd.to_numeric(sub[mc], errors="coerce").astype(np.float64).values
            vals = vals[np.isfinite(vals)]
            row[mc] = float(np.mean(vals)) if vals.size else float("nan")
        for bc in band_cols:
            # SEM across agents in the decade (based on their mean values)
            mc_paired = bc.replace("_band", "_mean")
            if mc_paired in sub.columns:
                vals = pd.to_numeric(sub[mc_paired], errors="coerce").astype(np.float64).values
                vals = vals[np.isfinite(vals)]
                row[bc] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            else:
                row[bc] = 0.0
        for ec in extra_cols:
            row[ec] = float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Conservative filter
# ---------------------------------------------------------------------------

def apply_conservative_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mantieni solo gli episodi (identificati da _file_idx + episode_idx + budget)
    in cui il conservative agent ha raggiunto il goal (success==1, oppure
    goal_first_step <= budget come fallback).  Le righe di tutti gli altri
    agenti per quegli stessi episodi vengono mantenute; le restanti eliminate.
    """
    if "conservative" not in df["agent"].astype(str).values:
        print("WARNING: agente 'conservative' non trovato – filtro conservative saltato.")
        return df

    if "success" in df.columns:
        cons_ok = (
            (df["agent"].astype(str) == "conservative")
            & (pd.to_numeric(df["success"], errors="coerce") == 1)
        )
    elif "goal_first_step" in df.columns and "budget" in df.columns:
        gf  = pd.to_numeric(df["goal_first_step"], errors="coerce")
        bud = pd.to_numeric(df["budget"],          errors="coerce")
        cons_ok = (
            (df["agent"].astype(str) == "conservative")
            & (gf != -1) & (gf <= bud)
        )
    else:
        print("WARNING: impossibile determinare il successo – filtro conservative saltato.")
        return df

    key_cols = ["_file_idx", "episode_idx", "budget"]
    success_keys = set(
        map(tuple, df.loc[cons_ok, key_cols].drop_duplicates().values.tolist())
    )
    n_total_cons = int((df["agent"].astype(str) == "conservative").sum())
    print(
        f"Conservative filter: {len(success_keys)} / {n_total_cons} episodi mantenuti "
        f"({100 * len(success_keys) / n_total_cons:.1f}% degli episodi conservative)."
    )

    mask = df[key_cols].apply(lambda r: tuple(r) in success_keys, axis=1)
    filtered = df[mask].copy()
    print(f"  Righe prima del filtro: {len(df)} → dopo: {len(filtered)}")
    return filtered


# ---------------------------------------------------------------------------
# CVaR helper
# ---------------------------------------------------------------------------

def cvar_worst_mean(costs: np.ndarray, alpha: float) -> float:
    """Mean of the worst ceil(N*alpha) costs (higher = worse)."""
    costs = np.asarray(costs, dtype=np.float64)
    costs = costs[np.isfinite(costs)]
    N = costs.size
    if N == 0:
        return float("nan")
    k = max(int(np.ceil(alpha * N)), 1)
    print(f"  CVaR: N={N}, alpha={alpha}, k={k}")
    return float(np.mean(np.sort(costs)[-k:]))


# ---------------------------------------------------------------------------
# Per-run computation
# ---------------------------------------------------------------------------

def compute_per_agent(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Per ogni agente nel DataFrame calcola mean_cost, cvar_cost, success_rate.
    Usa la colonna 'cost_total' per i costi e 'goal_first_step' per il successo.
    """
    if "agent" not in df.columns:
        raise KeyError("CSV manca della colonna 'agent'.")

    rows = []
    for agent, g in df.groupby("agent", sort=True):
        # --- costi ---
        if "cost_total" in g.columns:
            costs = pd.to_numeric(g["cost_total"], errors="coerce").astype(np.float64).values
        else:
            raise KeyError(
                f"Agente '{agent}': colonna 'cost_total' non trovata. "
                f"Colonne disponibili: {list(g.columns)}"
            )
        costs = costs[np.isfinite(costs)]

        # --- successo ---
        if "goal_first_step" in g.columns:
            gf = pd.to_numeric(g["goal_first_step"], errors="coerce").astype(np.float64).values
            success_rate = float(np.mean(gf != -1)) if gf.size else float("nan")
        elif "success" in g.columns:
            success_rate = float(pd.to_numeric(g["success"], errors="coerce").mean())
        else:
            success_rate = float("nan")

        rows.append({
            "agent":        agent,
            "n_episodes":   int(costs.size),
            "mean_cost":    float(np.mean(costs)) if costs.size else float("nan"),
            "cvar_cost":    cvar_worst_mean(costs, alpha),
            "success_rate": success_rate,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregazione multi-run
# ---------------------------------------------------------------------------

def aggregate_runs(run_dfs: list, band_mode: str) -> pd.DataFrame:
    """Aggrega più DataFrame (uno per run/seed) con media ± band."""
    all_df = pd.concat(
        [df.assign(run_id=i) for i, df in enumerate(run_dfs)],
        ignore_index=True,
    )

    def _band(arr):
        arr = arr[np.isfinite(arr)]
        if arr.size <= 1:
            return 0.0
        s = float(np.std(arr, ddof=1))
        return s if band_mode == "std" else s / float(np.sqrt(arr.size))

    out_rows = []
    for agent, g in all_df.groupby("agent", sort=True):
        mc = pd.to_numeric(g["mean_cost"],    errors="coerce").astype(np.float64).values
        cv = pd.to_numeric(g["cvar_cost"],    errors="coerce").astype(np.float64).values
        sr = pd.to_numeric(g["success_rate"], errors="coerce").astype(np.float64).values
        ne = pd.to_numeric(g["n_episodes"],   errors="coerce").astype(np.float64).values
        out_rows.append({
            "agent":              agent,
            "n_runs":             int(g["run_id"].nunique()),
            "n_episodes_mean":    float(np.nanmean(ne)),
            "mean_cost_mean":     float(np.nanmean(mc)),
            "mean_cost_band":     _band(mc),
            "cvar_cost_mean":     float(np.nanmean(cv)),
            "cvar_cost_band":     _band(cv),
            "success_rate_mean":  float(np.nanmean(sr)),
            "success_rate_band":  _band(sr),
        })
    result = pd.DataFrame(out_rows)
    # sort by penalty value (baselines first, then p10, p30, p100, ...)
    result = result.iloc[sorted(range(len(result)), key=lambda i: _penalty_sort_key(result.iloc[i]["agent"]))]
    result = result.reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_bar(agg_df: pd.DataFrame, metric_mean: str, metric_band: str,
             ylabel: str, title: str, out_path: str,
             label_map: dict = None, hline: float = None, hline_label: str = None):
    import matplotlib.ticker as mticker

    agents = agg_df["agent"].tolist()
    labels = [label_map.get(a, a) if label_map else a for a in agents]
    means  = agg_df[metric_mean].astype(float).values
    bands  = agg_df[metric_band].astype(float).values

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    agent_colors = [colors[i % len(colors)] for i in range(len(agents))]

    # Keep horizontal spacing consistent with the comparison plots.
    fig, ax = plt.subplots(figsize=(max(6, len(agents) * 0.9), 5))
    x = np.arange(len(agents))
    bar_width = 0.65

    rects = ax.bar(x, means, bar_width, yerr=bands,
                   color=agent_colors, capsize=5,
                   error_kw={"elinewidth": 1.5, "ecolor": "black", "capthick": 1.5},
                   edgecolor="white", linewidth=0.5)

    # Linea orizzontale (safety budget / limite)
    if hline is not None:
        lbl = hline_label if hline_label else f"Cost Limit = {hline:.4g}"
        ax.axhline(hline, color="red", linestyle="--", linewidth=1.4,
                   label=lbl, zorder=5)
        ax.legend(fontsize=13, loc="best")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato plot: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Calcola CVaR e media costi per agente (tutti gli episodi, nessuna suddivisione per budget)."
    )
    p.add_argument("--csvs", nargs="+", required=True,
                   help="Uno o più CSV (ognuno trattato come run/seed indipendente).")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Probabilità coda CVaR (default: 0.1 = peggiore 10%%).")
    p.add_argument("--band", choices=["se", "std"], default="se",
                   help="Banda di incertezza: errore standard (se, default) o std (std).")
    p.add_argument("--agents", nargs="+", default=None,
                   help="Sottoinsieme di agenti da includere (default: tutti).")
    p.add_argument("--agent_labels", nargs="+", default=None, metavar="NOME=ETICHETTA",
                   help="Rinomina agenti nei plot. Es: 'conservative=Conservative'.")
    p.add_argument("--out_csv", type=str, default="results/cvar_exp.csv")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Se specificato, i plot vengono salvati qui.")
    p.add_argument("--out_prefix", type=str, default="cvar",
                   help="Prefisso per i file plot (default: cvar).")
    p.add_argument("--cost_hline", type=float, default=None,
                   help="Valore del safety budget (linea rossa tratteggiata sui plot di costo).")
    p.add_argument("--cost_hline_label", type=str, default=None,
                   help="Etichetta per la linea hline (default: 'Limit (<valore>)').")
    p.add_argument("--by_decade", action="store_true",
                   help="Raggruppa gli agenti penalty in 3 decadi (p10-p100, p100-p1k, p1k-p10k) "
                        "e plotta la media per decade. I baseline vengono mostrati separatamente.")
    p.add_argument("--title", type=str, default="",
                   help="Titolo comune per tutti e 3 i plot (default: nessun titolo).")
    p.add_argument("--conservative_filter", action="store_true",
                   help="Mantieni solo gli episodi in cui il conservative agent ha raggiunto il goal. "
                        "Tutti gli altri agenti vengono valutati solo su quegli episodi condivisi.")
    args = p.parse_args()

    # label remap
    label_map = {}
    if args.agent_labels:
        for entry in args.agent_labels:
            eq = entry.rindex("=")
            label_map[entry[:eq]] = entry[eq + 1:]

    # carica CSV – ogni file riceve un _file_idx per evitare collisioni di episode_idx
    print(f"Caricamento {len(args.csvs)} CSV...")
    run_dfs_per_agent: dict[str, list] = {}
    for file_idx, path in enumerate(args.csvs):
        try:
            df = pd.read_csv(path)
            df["_file_idx"] = file_idx
        except Exception as e:
            print(f"  Warning: impossibile leggere {path}: {e}")
            continue

        if args.conservative_filter:
            df = apply_conservative_filter(df)
            if df.empty:
                print(f"  Skipping {path}: nessun episodio rimasto dopo il filtro.")
                continue

        if args.agents:
            df = df[df["agent"].astype(str).isin(args.agents)]

        per_agent_df = compute_per_agent(df, args.alpha)
        for _, row in per_agent_df.iterrows():
            run_dfs_per_agent.setdefault(row["agent"], []).append(
                pd.DataFrame([row])
            )

    if not run_dfs_per_agent:
        raise SystemExit("Nessun dato caricato. Controlla --csvs e --agents.")

    # aggrega run per ogni agente
    all_runs = [pd.concat(dfs, ignore_index=True)
                for dfs in run_dfs_per_agent.values()]
    # ogni run_df ha una riga per agente; rebuild a list of per-run full dfs
    n_runs = max(len(v) for v in run_dfs_per_agent.values())
    run_list = []
    for i in range(n_runs):
        rows = []
        for agent, dfs in run_dfs_per_agent.items():
            if i < len(dfs):
                rows.append(dfs[i])
        if rows:
            run_list.append(pd.concat(rows, ignore_index=True))

    agg = aggregate_runs(run_list, band_mode=args.band)

    if args.agents:
        available_agents = agg["agent"].astype(str).tolist()
        desired_order = [a for a in args.agents if a in available_agents]
        missing = [a for a in args.agents if a not in available_agents]
        if missing:
            print(f"WARNING: these agents were requested but not found in aggregated results: {missing}")

        remaining = [a for a in available_agents if a not in desired_order]
        agg = agg.set_index("agent").loc[desired_order + remaining].reset_index()

    # stampa risultati
    print(f"\nRisultati (alpha={args.alpha}, band={args.band}):\n")
    print(agg.to_string(index=False))

    # salva CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    agg.to_csv(args.out_csv, index=False)
    print(f"\nSalvato: {args.out_csv}")

    # plot
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        base = os.path.join(args.out_dir, args.out_prefix)

        plot_df = build_decade_agg(agg) if args.by_decade else agg

        plot_bar(plot_df, "mean_cost_mean", "mean_cost_band",
                 ylabel="Mean cost",
                 title=args.title,
                 out_path=f"{base}_mean_cost.png",
                 label_map=label_map,
                 hline=args.cost_hline, hline_label=args.cost_hline_label)

        plot_bar(plot_df, "cvar_cost_mean", "cvar_cost_band",
                 ylabel=f"CVaR_{args.alpha} (cost)",
                 title=args.title,
                 out_path=f"{base}_cvar_cost.png",
                 label_map=label_map,
                 hline=args.cost_hline, hline_label=args.cost_hline_label)

        plot_bar(plot_df, "success_rate_mean", "success_rate_band",
                 ylabel="Success rate",
                 title=args.title,
                 out_path=f"{base}_success_rate.png",
                 label_map=label_map)

        print(f"\nPlot salvati in: {args.out_dir}/")


if __name__ == "__main__":
    main()
