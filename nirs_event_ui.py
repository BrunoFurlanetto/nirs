"""
nirs_event_ui.py — Aba de análise por paradigma de EVENTO
==========================================================

No paradigma de bloco (aba Pipeline), a época começa no marcador e dura o bloco
inteiro. Aqui o marcador (letra) indica apenas o INÍCIO do bloco de 30 s; o
evento de interesse acontece X segundos depois do marcador. A época é recortada
em torno do evento: [evento - pré, evento + pós] (padrão ±10 s).

Reaproveita o arquivo carregado na aba Pipeline (eventos já mapeados, DPF e
canais excluídos), aplica Baseline initial time → Wavelet-MDL e extrai as
épocas centradas no evento.
"""

import io
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from pipeline_spm import (
    intensity_to_concentration,
    baseline_initial_time,
    wavelet_mdl,
    extract_epochs,
    FS,
)


def _preprocess(rec, dpf_values, keep_idx, baseline_samples, wavelet, wavelet_level):
    """Beer-Lambert (se intensidade) → exclusão de canais → baseline → wavelet."""
    if rec["mode"] == "intensity":
        dist_mm = [d for _, d, _ in st.session_state.distances]
        oxy, dxy = intensity_to_concentration(
            rec["intensity"], rec["measlist"], rec["wavelengths"],
            dist_mm, dpf_values,
        )
    else:
        oxy, dxy = rec["oxy"], rec["dxy"]

    oxy = oxy[:, keep_idx]
    dxy = dxy[:, keep_idx]
    oxy_bl, dxy_bl = baseline_initial_time(oxy, dxy, n_samples=int(baseline_samples))
    oxy_wav = wavelet_mdl(oxy_bl, wavelet=wavelet, level=int(wavelet_level))
    dxy_wav = wavelet_mdl(dxy_bl, wavelet=wavelet, level=int(wavelet_level))
    return oxy_wav, dxy_wav


def render(baseline_samples, wavelet, wavelet_level, state_prefix="event"):
    st.subheader("Análise por evento")
    st.caption(
        "O marcador (letra) indica o **início do bloco**. Informe em que momento, "
        "depois do marcador, o evento aconteceu. A época é recortada em torno do "
        "evento: de **evento − pré** até **evento + pós**."
    )

    if not st.session_state.get("loaded"):
        st.info("Carregue um arquivo na aba **Pipeline** para começar.")
        return

    rec = st.session_state.rec
    events = st.session_state.events
    n_samples = rec["n_samples"]
    duration_s = n_samples / FS
    filename = st.session_state.filename
    ch_labels = st.session_state.get("pipe_ch_labels") or [
        f"Ch{i+1}" for i in range(st.session_state.n_ch)
    ]
    keep_idx = st.session_state.get("pipe_keep_idx")
    if keep_idx is None:
        keep_idx = list(range(st.session_state.n_ch))
    dpf_values = st.session_state.get("pipe_dpf_values")

    if not events:
        st.warning("O arquivo não tem eventos.")
        return
    if not keep_idx:
        st.error("Todos os canais foram excluídos na aba Pipeline — a análise não pode rodar.")
        return
    if rec["mode"] == "intensity" and dpf_values is None:
        st.warning("Abra a aba **Pipeline** uma vez para definir o DPF antes de rodar.")
        return

    st.caption(
        f"Arquivo: **{filename}** — {len(keep_idx)} canais em uso "
        "(exclusão e DPF definidos na aba Pipeline)."
    )

    # ── 1. Evento e janela ────────────────────────────────────────────────────
    st.markdown("**1. Evento e janela**")
    c1, c2 = st.columns([2, 1])
    with c1:
        cond = st.selectbox(
            "Evento (marcador) a analisar",
            list(events.keys()),
            key=f"{state_prefix}_cond",
            help="Condição cujo marcador indica o início do bloco.",
        )
    with c2:
        block_dur = st.number_input(
            "Duração do bloco (s)", min_value=1.0, max_value=600.0, value=30.0,
            step=1.0, key=f"{state_prefix}_block",
        )

    c3, c4, c5 = st.columns(3)
    with c3:
        offset = st.number_input(
            "Tempo do evento após o marcador (s)", min_value=0.0,
            max_value=float(block_dur), value=0.0, step=0.1,
            key=f"{state_prefix}_offset",
            help="Momento do evento dentro do bloco, contado a partir do marcador. "
                 "Pode ser ajustado por ocorrência na tabela abaixo.",
        )
    with c4:
        pre = st.number_input(
            "Tempo antes do evento (s)", min_value=0.1, max_value=300.0,
            value=10.0, step=0.5, key=f"{state_prefix}_pre",
        )
    with c5:
        post = st.number_input(
            "Tempo depois do evento (s)", min_value=0.1, max_value=300.0,
            value=10.0, step=0.5, key=f"{state_prefix}_post",
        )

    c6, c7 = st.columns(2)
    with c6:
        bl_start = st.number_input(
            "Baseline da época — início (s)", min_value=-float(pre), max_value=float(post),
            value=-min(2.0, float(pre)), step=0.5, key=f"{state_prefix}_bl_start",
            help="Relativo ao evento (negativo = antes do evento).",
        )
    with c7:
        bl_end = st.number_input(
            "Baseline da época — fim (s)", min_value=-float(pre), max_value=float(post),
            value=0.0, step=0.5, key=f"{state_prefix}_bl_end",
        )
    if bl_end <= bl_start:
        st.warning("Baseline fim ≤ início — a correção de baseline não será aplicada.")

    # ── 2. Tempo do evento por ocorrência ─────────────────────────────────────
    st.markdown("**2. Tempo do evento por ocorrência**")
    st.caption(
        "Por padrão todas as ocorrências usam o tempo informado acima. "
        "Edite a coluna para ajustar ocorrências individuais; desmarque para excluir."
    )
    onsets = sorted(float(o) for o in events[cond])
    df_occ = pd.DataFrame({
        "Ocorrência": list(range(1, len(onsets) + 1)),
        "Marcador (s)": [round(o, 2) for o in onsets],
        "Tempo após marcador (s)": [float(offset)] * len(onsets),
        "Usar": [True] * len(onsets),
    })
    # A key inclui condição e offset global: trocar qualquer um reseta a tabela
    # para os novos padrões em vez de manter edições de outra configuração.
    df_occ_edit = st.data_editor(
        df_occ,
        column_config={
            "Ocorrência": st.column_config.NumberColumn(disabled=True),
            "Marcador (s)": st.column_config.NumberColumn(disabled=True, format="%.2f"),
            "Tempo após marcador (s)": st.column_config.NumberColumn(
                min_value=0.0, max_value=float(block_dur), step=0.1, format="%.2f",
            ),
            "Usar": st.column_config.CheckboxColumn(),
        },
        hide_index=True,
        use_container_width=True,
        key=f"{state_prefix}_occ_{cond}_{offset}_{block_dur}",
    )

    event_times = [
        float(m) + float(off)
        for m, off, use in zip(df_occ_edit["Marcador (s)"],
                               df_occ_edit["Tempo após marcador (s)"],
                               df_occ_edit["Usar"])
        if use
    ]
    # Evento fora do registro não gera época; descartar aqui mantém o índice
    # de event_times alinhado com as épocas devolvidas por extract_epochs.
    descartados = [t for t in event_times if not 0 <= t < duration_s]
    if descartados:
        st.warning("Ocorrência(s) fora do registro descartada(s): "
                   + ", ".join(f"{t:.1f}s" for t in descartados))
        event_times = [t for t in event_times if 0 <= t < duration_s]
    fora = [t for t in event_times if t - pre < 0 or t + post > duration_s]
    if fora:
        st.warning(
            f"{len(fora)} ocorrência(s) com janela fora do sinal "
            f"({', '.join(f'{t:.1f}s' for t in fora)}) — o trecho faltante é preenchido com zero."
        )
    if event_times:
        st.caption("Evento em: " + ", ".join(f"{t:.1f}s" for t in event_times))

    # ── 3. Rodar ──────────────────────────────────────────────────────────────
    st.markdown("**3. Rodar**")
    if not st.button("▶ Rodar análise por evento", type="primary",
                     disabled=not event_times, key=f"{state_prefix}_run"):
        return

    # Pré-processamento só depende do arquivo e dos parâmetros globais: guarda em
    # cache para que mudar a janela/tempo do evento não refaça o Wavelet-MDL.
    dpf_key = tuple(sorted(dpf_values.items())) if dpf_values else None
    cache_key = (st.session_state.get("file_hash") or filename, tuple(keep_idx),
                 dpf_key, int(baseline_samples), wavelet, int(wavelet_level))
    cache = st.session_state.get(f"{state_prefix}_prep")
    if cache is None or cache[0] != cache_key:
        with st.spinner("Baseline initial time → Wavelet-MDL (pode demorar)..."):
            oxy_wav, dxy_wav = _preprocess(rec, dpf_values, keep_idx,
                                           baseline_samples, wavelet, wavelet_level)
        st.session_state[f"{state_prefix}_prep"] = (cache_key, oxy_wav, dxy_wav)
    else:
        _, oxy_wav, dxy_wav = cache

    ev = {cond: event_times}
    kwargs = dict(tmin=-float(pre), fs=FS, baseline=(float(bl_start), float(bl_end)),
                  durations={cond: float(post)}, post_margin=0.0)
    epochs_oxy, _ = extract_epochs(oxy_wav, ev, **kwargs)
    epochs_dxy, _ = extract_epochs(dxy_wav, ev, **kwargs)

    if cond not in epochs_oxy or cond not in epochs_dxy:
        st.error("Nenhuma época válida extraída.")
        return

    times = epochs_oxy[cond]["times"]
    n_trials = epochs_oxy[cond]["n"]
    kept_labels = [ch_labels[i] for i in keep_idx]
    ch_names = [lbl.split(" ")[0] for lbl in kept_labels]

    hbo = np.mean(epochs_oxy[cond]["mean"], axis=1)
    hbr = np.mean(epochs_dxy[cond]["mean"], axis=1)
    hbo_sem = np.mean(epochs_oxy[cond]["sem"], axis=1)
    hbr_sem = np.mean(epochs_dxy[cond]["sem"], axis=1)

    st.success(f"Análise concluída — {n_trials} ocorrência(s) de **{cond}**.")

    # ── Gráfico médio ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.6))
    if n_trials > 1:
        ax.fill_between(times, hbo - hbo_sem, hbo + hbo_sem, color="#d62728", alpha=0.2)
        ax.fill_between(times, hbr - hbr_sem, hbr + hbr_sem, color="#1f77b4", alpha=0.2)
    ax.plot(times, hbo, color="#d62728", label="HbO", linewidth=1.6)
    ax.plot(times, hbr, color="#1f77b4", label="HbR", linewidth=1.6)
    ax.axvline(0, color="black", linestyle="--", alpha=0.6, label="Evento")
    ax.axhline(0, color="gray", alpha=0.3)
    if bl_end > bl_start:
        ax.axvspan(bl_start, bl_end, color="gray", alpha=0.1, label="Baseline")
    ax.set_title(f"{cond} — evento (n={n_trials}, −{pre:g}s / +{post:g}s)", fontsize=10)
    ax.set_xlabel("Tempo relativo ao evento (s)", fontsize=8)
    ax.set_ylabel("µM", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    col_plot, _ = st.columns([2, 1])
    col_plot.pyplot(fig)
    plt.close(fig)

    # ── Gráfico por canal ─────────────────────────────────────────────────────
    with st.expander("Resposta por canal"):
        oxy_ch = epochs_oxy[cond]["mean"]
        dxy_ch = epochs_dxy[cond]["mean"]
        ncols = 4
        for start in range(0, len(kept_labels), ncols):
            cols = st.columns(ncols)
            for cw, ci in zip(cols, range(start, min(start + ncols, len(kept_labels)))):
                f, a = plt.subplots(figsize=(4, 2.4))
                a.plot(times, oxy_ch[:, ci], color="#d62728", linewidth=1.0, label="HbO")
                a.plot(times, dxy_ch[:, ci], color="#1f77b4", linewidth=1.0, label="HbR")
                a.axvline(0, color="black", linestyle="--", alpha=0.6)
                a.set_title(kept_labels[ci], fontsize=9)
                a.tick_params(labelsize=6)
                a.grid(True, alpha=0.3)
                a.legend(fontsize=6)
                cw.pyplot(f)
                plt.close(f)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("**Download**")
    mean_rows, ch_rows, trial_rows = [], [], []
    oxy_ch = epochs_oxy[cond]["mean"]
    dxy_ch = epochs_dxy[cond]["mean"]
    for ti, t_val in enumerate(times):
        t_r = round(float(t_val), 2)
        mean_rows.append({
            "Condicao": cond, "Tempo_s": t_r,
            "HbO_uM": float(hbo[ti]), "HbR_uM": float(hbr[ti]),
            "HbO_SEM": float(hbo_sem[ti]), "HbR_SEM": float(hbr_sem[ti]),
            "N_trials": n_trials,
        })
        row = {"Condicao": cond, "Tempo_s": t_r}
        for ci, ch in enumerate(ch_names):
            row[f"{ch}_HbO"] = float(oxy_ch[ti, ci])
            row[f"{ch}_HbR"] = float(dxy_ch[ti, ci])
        ch_rows.append(row)

    for trial_idx, (eo, ed) in enumerate(zip(epochs_oxy[cond]["epochs"],
                                             epochs_dxy[cond]["epochs"])):
        for ti, t_val in enumerate(times):
            row = {"Condicao": cond, "Trial": trial_idx + 1,
                   "Evento_s": round(event_times[trial_idx], 2),
                   "Tempo_s": round(float(t_val), 2)}
            for ci, ch in enumerate(ch_names):
                row[f"{ch}_HbO"] = float(eo[ti, ci])
                row[f"{ch}_HbR"] = float(ed[ti, ci])
            trial_rows.append(row)

    stem = f"{os.path.splitext(filename)[0]}_{cond}_evento"
    d1, d2, d3 = st.columns(3)
    for col, rows, suffix, label in (
        (d1, mean_rows, "media", "Média entre canais (.csv)"),
        (d2, ch_rows, "canais", "Por canal (.csv)"),
        (d3, trial_rows, "trials", "Ocorrências individuais (.csv)"),
    ):
        buf = io.StringIO()
        pd.DataFrame(rows).to_csv(buf, index=False, sep=";")
        col.download_button(label, buf.getvalue().encode("utf-8"),
                            file_name=f"{stem}_{suffix}.csv", mime="text/csv",
                            key=f"{state_prefix}_dl_{suffix}")
