"""
analise_janelada.py — Análise janelada de dados fNIRS (saída _hrf_canais.csv)

Uso:
    python analise_janelada.py <caminho_csvs> <saida.csv> [--condicoes C1 C2 ...]

Argumentos:
    caminho_csvs  Pasta contendo os arquivos _hrf_canais.csv (busca recursiva)
    saida.csv     Caminho do arquivo de saída
    --condicoes   (opcional) Lista de condições a processar; sem filtro = todas
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Análise janelada de HbO/HbR por condição")
    p.add_argument("caminho_csvs", type=Path, help="Pasta com os CSVs de entrada")
    p.add_argument("saida", type=Path, help="Caminho do CSV de saída")
    p.add_argument("--condicoes", nargs="*", default=None, metavar="COND",
                   help="Condições a processar (padrão: todas)")
    return p.parse_args()


def find_csvs(folder: Path) -> list[Path]:
    return sorted(folder.rglob("*.csv"))


def hbo_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_HbO")]


def hbr_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_HbR")]


def window_means(series_values: np.ndarray, times: np.ndarray, n_windows: int):
    """Divide [times] em n_windows iguais e retorna a média de cada janela."""
    t_min, t_max = times.min(), times.max()
    duration = t_max - t_min
    win_size = duration / n_windows
    means = []
    for i in range(n_windows):
        t_start = t_min + i * win_size
        t_end = t_min + (i + 1) * win_size
        if i == n_windows - 1:
            mask = (times >= t_start) & (times <= t_end)
        else:
            mask = (times >= t_start) & (times < t_end)
        means.append(float(np.mean(series_values[mask])) if mask.any() else np.nan)
    return means, win_size


def process_file_condition(df: pd.DataFrame, condition: str, filename: str) -> dict | None:
    sub = df[df["Condicao"] == condition].copy()
    if sub.empty:
        return None

    hbo = hbo_cols(df)
    hbr = hbr_cols(df)
    if not hbo or not hbr:
        print(f"  [AVISO] {filename}: colunas HbO/HbR não encontradas", file=sys.stderr)
        return None

    # Baseline: Tempo_s < 0, excluindo os últimos 5s (Tempo_s < -5)
    bl = sub[sub["Tempo_s"] < -5]
    if bl.empty:
        print(f"  [AVISO] {filename} / {condition}: baseline vazio após corte de 5s", file=sys.stderr)
        return None

    baseline_hbo = float(bl[hbo].values.mean())
    baseline_hbr = float(bl[hbr].values.mean())

    # Tarefa: Tempo_s >= 0, excluindo os primeiros 5s (Tempo_s >= 5)
    task = sub[sub["Tempo_s"] >= 5]
    if task.empty:
        print(f"  [AVISO] {filename} / {condition}: tarefa vazia após corte de 5s", file=sys.stderr)
        return None

    task_times = task["Tempo_s"].values
    task_duration = task_times.max() - task_times.min()
    n_windows = max(1, math.ceil(task_duration / 30))

    # Média geral de todos os canais por ponto de tempo → série 1D
    hbo_series = task[hbo].values.mean(axis=1)
    hbr_series = task[hbr].values.mean(axis=1)

    hbo_wins, win_size = window_means(hbo_series, task_times, n_windows)
    hbr_wins, _ = window_means(hbr_series, task_times, n_windows)

    media_hbo = float(np.nanmean(hbo_wins))
    media_hbr = float(np.nanmean(hbr_wins))

    row = {
        "arquivo": filename,
        "condicao": condition,
        "n_janelas": n_windows,
        "tamanho_janela_s": round(win_size, 4),
        "baseline_HbO": baseline_hbo,
        "baseline_HbR": baseline_hbr,
    }
    for i, (h, r) in enumerate(zip(hbo_wins, hbr_wins), start=1):
        row[f"janela_{i}_HbO"] = h
        row[f"janela_{i}_HbR"] = r
    row["media_janelas_HbO"] = media_hbo
    row["media_janelas_HbR"] = media_hbr

    return row


def main():
    args = parse_args()

    csvs = find_csvs(args.caminho_csvs)
    if not csvs:
        sys.exit(f"Nenhum CSV encontrado em: {args.caminho_csvs}")

    print(f"{len(csvs)} arquivo(s) encontrado(s).")

    all_rows: list[dict] = []

    for csv_path in csvs:
        print(f"Lendo: {csv_path.name}")
        try:
            df = pd.read_csv(csv_path, sep=";", decimal=".")
        except Exception as e:
            print(f"  [ERRO] Falha ao ler {csv_path.name}: {e}", file=sys.stderr)
            continue

        if "Condicao" not in df.columns or "Tempo_s" not in df.columns:
            print(f"  [AVISO] {csv_path.name}: colunas 'Condicao'/'Tempo_s' ausentes — pulando",
                  file=sys.stderr)
            continue

        conditions = args.condicoes if args.condicoes else sorted(df["Condicao"].unique())

        for cond in conditions:
            if cond not in df["Condicao"].values:
                print(f"  [AVISO] {csv_path.name}: condição '{cond}' não encontrada — pulando",
                      file=sys.stderr)
                continue
            row = process_file_condition(df, cond, csv_path.name)
            if row is not None:
                all_rows.append(row)

    if not all_rows:
        sys.exit("Nenhum dado processado. Verifique os arquivos e condições informados.")

    result = pd.DataFrame(all_rows)

    # Garante ordem das colunas: fixas + janelas ordenadas + totais
    fixed_cols = ["arquivo", "condicao", "n_janelas", "tamanho_janela_s",
                  "baseline_HbO", "baseline_HbR"]
    max_wins = max(r["n_janelas"] for r in all_rows)
    win_cols = []
    for i in range(1, max_wins + 1):
        win_cols += [f"janela_{i}_HbO", f"janela_{i}_HbR"]
    tail_cols = ["media_janelas_HbO", "media_janelas_HbR"]

    ordered = fixed_cols + win_cols + tail_cols
    result = result.reindex(columns=ordered)

    saida = args.saida
    if saida.is_dir() or not saida.suffix:
        saida = saida / "analise_janelada.csv"
    saida.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(saida, sep=";", index=False, decimal=".")
    print(f"\nSalvo em: {saida}")
    print(f"Linhas: {len(result)} | Janelas máximas: {max_wins}")


if __name__ == "__main__":
    main()
