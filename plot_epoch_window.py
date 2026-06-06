"""
plot_epoch_window.py
--------------------
Gera figura de publicação do HRF com as janelas de análise desenhadas.

Uso:
    python plot_epoch_window.py hrf_canais.csv COND --analise analise_janelada.csv --arquivo dp340_pre_cog2_hrf_canais.csv
    python plot_epoch_window.py hrf_canais.csv S --analise analise_janelada.csv --arquivo dp340_pre_cog2_hrf_canais.csv --channel Ch3
    python plot_epoch_window.py hrf_canais.csv S --analise analise_janelada.csv --arquivo dp340_pre_cog2_hrf_canais.csv --output fig.pdf
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "lines.linewidth":    1.4,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

HBO_COLOR  = "#c0392b"
HBR_COLOR  = "#2980b9"

WINDOW_PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]

ONSET_SKIP    = 5.0   # segundos iniciais do onset excluídos
BASELINE_SKIP = 5.0   # segundos finais do baseline excluídos


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("csv_file",  help="CSV hrf_canais (Condicao, Tempo_s, Ch*_HbO, Ch*_HbR)")
    p.add_argument("condition", help="Nome da condição (ex: S)")
    p.add_argument("--analise", default=None, metavar="analise_janelada.csv",
                   help="CSV analise_janelada com as janelas de análise.")
    p.add_argument("--arquivo", default=None,
                   help="Nome do arquivo na coluna 'arquivo' do CSV analise_janelada "
                        "(ex: dp340_pre_cog2_hrf_canais.csv).")
    p.add_argument("--channel", default=None,
                   help="Canal específico (ex: Ch3). Padrão: média de todos.")
    p.add_argument("--output", default=None)
    p.add_argument("--dpi",    type=int,   default=300)
    p.add_argument("--width",  type=float, default=3.5)
    p.add_argument("--height", type=float, default=2.8)
    return p.parse_args()


def load_hrf(csv_path, condition, channel):
    df = pd.read_csv(csv_path, sep=";", decimal=".")
    cond_df = df[df["Condicao"] == condition]
    if cond_df.empty:
        avail = df["Condicao"].unique().tolist()
        sys.exit(f"Condição '{condition}' não encontrada. Disponíveis: {avail}")

    times    = cond_df["Tempo_s"].values
    hbo_cols = [c for c in cond_df.columns if c.endswith("_HbO")]
    hbr_cols = [c for c in cond_df.columns if c.endswith("_HbR")]

    if channel:
        hc, dc = f"{channel}_HbO", f"{channel}_HbR"
        if hc not in cond_df.columns:
            avail_ch = [c.replace("_HbO", "") for c in hbo_cols]
            sys.exit(f"Canal '{channel}' não encontrado. Disponíveis: {avail_ch}")
        hbo, hbr = cond_df[hc].values, cond_df[dc].values
        label_ch = channel
    else:
        hbo      = cond_df[hbo_cols].values.mean(axis=1)
        hbr      = cond_df[hbr_cols].values.mean(axis=1)
        label_ch = f"média ({len(hbo_cols)} canais)"

    return times, hbo, hbr, label_ch


def load_janelas(analise_path, arquivo, condition):
    df = pd.read_csv(analise_path, sep=";", decimal=".")
    mask = (df["condicao"] == condition)
    if arquivo:
        mask &= (df["arquivo"] == arquivo)
    row = df[mask]
    if row.empty:
        sys.exit(f"Condição '{condition}' / arquivo '{arquivo}' não encontrado em {analise_path}.")
    row = row.iloc[0]
    n       = int(row["n_janelas"])
    tam     = float(row["tamanho_janela_s"])
    bl_hbo  = float(row["baseline_HbO"])
    bl_hbr  = float(row["baseline_HbR"])
    janelas = []
    for i in range(1, n + 1):
        janelas.append({
            "HbO": float(row[f"janela_{i}_HbO"]),
            "HbR": float(row[f"janela_{i}_HbR"]),
        })
    return n, tam, bl_hbo, bl_hbr, janelas


def main():
    args = parse_args()

    times, hbo, hbr, label_ch = load_hrf(args.csv_file, args.condition, args.channel)
    tmin, tmax = times[0], times[-1]

    janelas_info = None
    if args.analise:
        n_jan, tam, bl_hbo, bl_hbr, janelas_info = load_janelas(
            args.analise, args.arquivo, args.condition
        )

    fig, ax = plt.subplots(figsize=(args.width, args.height))

    # Linha de onset e zero
    ax.axvline(0, color="gray", lw=0.8, linestyle="--", alpha=0.5, zorder=2)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.3, zorder=2)

    # Janelas de análise como retângulos coloridos
    if janelas_info:
        # Baseline: de tmin até -BASELINE_SKIP
        bl_start = tmin
        bl_end   = -BASELINE_SKIP
        ax.axvspan(bl_start, bl_end, alpha=0.12, color="#7f8c8d", lw=0, zorder=1)

        # Janelas de tarefa: começam em ONSET_SKIP
        w_start = ONSET_SKIP
        for i, jan in enumerate(janelas_info):
            w_end = w_start + tam
            color = WINDOW_PALETTE[i % len(WINDOW_PALETTE)]
            ax.axvspan(w_start, w_end, alpha=0.18, color=color, lw=0, zorder=1)
            ax.axvline(w_start, color=color, lw=0.6, alpha=0.7, zorder=2)
            ax.axvline(w_end,   color=color, lw=0.6, alpha=0.7, zorder=2)
            w_start = w_end

    # Curvas HRF
    ax.plot(times, hbo, color=HBO_COLOR, lw=1.6, zorder=5)
    ax.plot(times, hbr, color=HBR_COLOR, lw=1.6, zorder=5)

    ax.set_xlabel("Tempo relativo ao onset (s)")
    ax.set_ylabel("Δ Concentração (µM)")
    ax.set_xlim(tmin, tmax)

    # Legenda
    legend_handles = [
        Line2D([0], [0], color=HBO_COLOR, lw=1.6, label="HbO"),
        Line2D([0], [0], color=HBR_COLOR, lw=1.6, label="HbR"),
    ]
    if janelas_info:
        legend_handles.append(
            mpatches.Patch(color="#7f8c8d", alpha=0.3, label="Baseline")
        )

    ax.legend(handles=legend_handles, frameon=False, loc="upper left",
              handlelength=1.0, handletextpad=0.4, labelspacing=0.3)

    ax.set_title(f"Condição {args.condition} — {label_ch}", fontsize=10)
    fig.tight_layout()

    out_path = args.output or f"{args.condition}_epoch.pdf"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva em: {out_path}")


if __name__ == "__main__":
    main()
