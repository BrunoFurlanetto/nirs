"""
nirs_concat_ui.py
=================

Interface Streamlit para concatenar arquivos .nirs de uma mesma sessão.

Feita para ser embutida no app principal como uma aba:

    import streamlit as st
    from nirs_concat_ui import render as render_merge

    tabs = st.tabs(["Merge", "Raw Data", "Probe", "QC", "HRF", "Convert"])
    with tabs[0]:
        render_merge(on_complete=minha_funcao_de_handoff)

O callback `on_complete(payload)` recebe:

    {
        "path":        Path do .nirs concatenado em disco,
        "result":      MergeResult (dados já em memória, sem precisar reler),
        "block_info":  dict com limites dos blocos (1-based, pronto p/ MATLAB),
        "sidecars":    dict com os caminhos dos arquivos auxiliares,
    }

Se `on_complete` não for passado, o resultado fica em
`st.session_state[f"{state_prefix}_payload"]` para o app principal consumir.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import streamlit as st

import nirs_concat_core as core

__all__ = ["render"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _natural_key(name: str):
    """Ordena bloco2 antes de bloco10, ao contrário da ordem alfabética."""
    return [int(p) if p.isdigit() else p.lower()
            for p in re.split(r"(\d+)", name)]


def _persist_uploads(uploaded, workdir: Path) -> list[Path]:
    """Grava os uploads em disco — o core lê de caminho, não de buffer."""
    workdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for uf in uploaded:
        p = workdir / uf.name
        p.write_bytes(uf.getbuffer())
        paths.append(p)
    return paths


def _fmt_dur(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}min {s:02d}s" if m else f"{s}s"


# --------------------------------------------------------------------------
# Blocos da UI
# --------------------------------------------------------------------------

def _section_order(files: list[core.NirsFile], key: str) -> list[core.NirsFile]:
    """Define a ordem cronológica. Nada mais importa se essa ordem estiver errada."""
    st.markdown("**Ordem de gravação**")
    st.caption(
        "O eixo de tempo do arquivo final é reconstruído nesta ordem. "
        "Arraste os itens para definir a sequência correta antes de juntar."
    )

    try:
        from streamlit_sortables import sort_items
    except ImportError:
        st.warning("Instale `streamlit-sortables` para arrastar e reordenar.")
        return files

    default_order = sorted(range(len(files)), key=lambda i: _natural_key(files[i].name))
    default_labels = [files[i].name for i in default_order]

    # Persist order across reruns; reset if file set changes.
    ss_key = f"{key}_order_list"
    stored: list[str] = st.session_state.get(ss_key, [])
    if set(stored) != set(default_labels):
        stored = default_labels

    ordered_labels: list[str] = sort_items(stored, direction="vertical", key=f"{key}_order")
    st.session_state[ss_key] = ordered_labels

    by_name = {f.name: f for f in files}
    return [by_name[n] for n in ordered_labels]


def _section_files(files: list[core.NirsFile]) -> None:
    rows = [
        {
            "arquivo": f.name,
            "amostras": f.n_samples,
            "colunas": f.n_cols,
            "fs (Hz)": round(f.fs, 2),
            "duração": _fmt_dur(f.duration_s),
            "condições": ", ".join(f.cond_names) or "—",
        }
        for f in files
    ]
    st.dataframe(rows, hide_index=True, width="stretch")


def _section_diagnostics(res: core.MergeResult) -> None:
    """Mostra o degrau medido em cada emenda, com leitura do que ele sugere."""
    if not res.diagnostics:
        return

    st.markdown("**Degrau nas emendas**")
    st.caption(
        "Medido no sinal original, antes de qualquer correção. Compara a mediana "
        "dos últimos segundos de um bloco com os primeiros do seguinte, canal a canal."
    )

    for dg in res.diagnostics:
        if dg.n_cols_over_threshold == 0:
            icon = "🟢"
        elif dg.some_channels_shifted:
            icon = "🔴"
        else:
            icon = "🟡"

        st.markdown(
            f"{icon} **Emenda {dg.index}** em {dg.time_s:.1f}s — "
            f"maior variação {dg.max_abs_pct:.1f}%, "
            f"{dg.n_cols_over_threshold}/{dg.n_cols} canais acima de {dg.threshold_pct:.0f}%"
        )
        st.caption(dg.verdict())

        if dg.some_channels_shifted:
            worst = np.argsort(-np.abs(dg.pct_change))[:5]
            st.caption(
                "Canais mais afetados: "
                + ", ".join(f"col {int(i) + 1} ({dg.pct_change[int(i)]:+.1f}%)" for i in worst)
            )


def _section_preview(res_raw: core.MergeResult,
                     res_final: core.MergeResult,
                     key: str) -> None:
    """Sinal antes e depois, com as emendas marcadas."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.info("Instale `plotly` para ver a prévia do sinal.")
        return

    col = st.selectbox(
        "Coluna para inspeção",
        options=list(range(res_raw.n_cols)),
        format_func=lambda i: f"Coluna {i + 1}",
        key=f"{key}_preview_col",
    )

    EVENT_PALETTE = ["#38a169", "#805ad5", "#d69e2e", "#0987a0",
                     "#d53f8c", "#c05621", "#2f855a", "#4a5568"]

    t = res_raw.t[:, 0]
    y = res_final.d[:, col]
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t, y=y, name="sinal concatenado",
        line=dict(width=1.1, color="#2b6cb0"),
    ))

    # Eventos — um trace por condição, None como separador de segmentos
    if res_final.s.size > 0:
        for i, name in enumerate(res_final.cond_names):
            idx = np.where(res_final.s[:, i] > 0)[0]
            if not idx.size:
                continue
            color = EVENT_PALETTE[i % len(EVENT_PALETTE)]
            xs, ys = [], []
            for sample in idx:
                x_ev = float(t[sample]) if sample < len(t) else sample / res_final.fs
                xs.extend([x_ev, x_ev, None])
                ys.extend([y_min, y_max, None])
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name=name,
                line=dict(width=1.0, color=color),
                hoverinfo="skip",
            ))

    for jt in res_raw.joint_times:
        fig.add_vline(x=jt, line_dash="dash", line_color="#e53e3e", line_width=1)

    has_events = res_final.s.size > 0 and any(
        np.where(res_final.s[:, i] > 0)[0].size for i in range(res_final.s.shape[1])
    )
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Tempo (s)",
        yaxis_title="Intensidade",
        showlegend=has_events,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
    )
    st.plotly_chart(fig, width="stretch")
    cap = "Linhas vermelhas tracejadas marcam as emendas."
    if has_events:
        cap += " Linhas coloridas marcam os eventos, uma cor por condição."
    st.caption(cap)


# --------------------------------------------------------------------------
# Entrada principal
# --------------------------------------------------------------------------

def render(on_complete: Callable[[dict], None] | None = None,
           state_prefix: str = "nirsmerge",
           default_trim_s: float = 5.0,
           default_baseline_s: float = 10.0,
           workdir: Path | str | None = None) -> dict | None:
    """
    Desenha a aba de concatenação.

    Parameters
    ----------
    on_complete : callback chamado quando o usuário confirma a junção.
    state_prefix : prefixo das chaves em st.session_state, para não colidir
        com o resto do app.
    default_trim_s : corte inicial padrão, em segundos.
    default_baseline_s : janela de baseline. Deve bater com o P.base da
        pipeline para que a equivalência algébrica seja exata.
    workdir : onde gravar os arquivos. Padrão: diretório temporário.

    Returns
    -------
    O payload da junção, ou None se ainda não houver resultado.
    """
    k = state_prefix
    workdir = Path(workdir) if workdir else Path(tempfile.gettempdir()) / k
    workdir.mkdir(parents=True, exist_ok=True)

    st.subheader("Juntar arquivos da mesma sessão")
    st.caption(
        "Para sessões que foram gravadas em blocos separados — reinício do software, "
        "pausa entre tarefas. Pressupõe que nenhuma tarefa atravessa dois arquivos."
    )

    uploaded = st.file_uploader(
        "Arquivos .nirs",
        type=["nirs"],
        accept_multiple_files=True,
        key=f"{k}_upload",
    )

    if not uploaded:
        st.info("Envie dois ou mais arquivos .nirs da mesma sessão.")
        return st.session_state.get(f"{k}_payload")

    if len(uploaded) < 2:
        st.warning("São necessários pelo menos dois arquivos para concatenar.")
        return None

    # ---------------------------------------------------------------- carregar
    paths = _persist_uploads(uploaded, workdir / "input")
    files: list[core.NirsFile] = []
    for p in paths:
        try:
            files.append(core.load_nirs(p))
        except Exception as exc:
            st.error(f"**{p.name}** — {exc}")
            return None

    _section_files(files)

    # ---------------------------------------------------------------- validar
    errors, warns = core.check_compatibility(files)
    for w in warns:
        st.warning(w)
    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    # ---------------------------------------------------------------- ordem
    ordered = _section_order(files, k)
    if not ordered:
        return None

    # ---------------------------------------------------------------- opções
    st.markdown("**Opções**")
    c1, c2 = st.columns(2)

    with c1:
        trim_s = st.number_input(
            "Cortar do início de cada bloco (s)",
            min_value=0.0, max_value=60.0,
            value=float(default_trim_s), step=1.0,
            key=f"{k}_trim",
            help="Descarta o transiente inicial de cada gravação.",
        )

    with c2:
        baseline_s = st.number_input(
            "Janela de baseline (s)",
            min_value=1.0, max_value=120.0,
            value=float(default_baseline_s), step=1.0,
            key=f"{k}_baseline",
            help=(
                "Precisa bater com o P.base da pipeline. É o que torna a conversão "
                "global equivalente a converter bloco a bloco."
            ),
        )

    reference = core.REF_BASELINE

    # ---------------------------------------------------------------- executar
    try:
        # O merge cru serve só para medir o degrau original — é ele que
        # distingue "capacete moveu" de "ganho recalibrado". Não vai para disco.
        res_raw = core.merge_nirs(ordered, mode=core.MODE_RAW, trim_s=trim_s)
        res_final = core.merge_nirs(ordered, mode=core.MODE_CORRECTED, trim_s=trim_s,
                                    reference=reference, baseline_s=baseline_s)
    except ValueError as exc:
        st.error(str(exc))
        return None

    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Blocos", len(ordered))
    m2.metric("Amostras", f"{res_final.n_samples:,}".replace(",", "."))
    m3.metric("Duração", _fmt_dur(res_final.duration_s))
    m4.metric("fs", f"{res_final.fs:.1f} Hz")

    counts = res_final.event_counts()
    if counts:
        st.markdown("**Eventos** — " + " · ".join(f"{n}: {c}" for n, c in counts.items()))

    with st.expander("Blocos no arquivo final", expanded=False):
        st.dataframe(res_final.block_table(), hide_index=True, width="stretch")
        st.caption(
            "Índices em base 1, prontos para uso no MATLAB. Use-os para aplicar o "
            "detrending bloco a bloco e para montar os regressores de bloco no GLM."
        )

    _section_diagnostics(res_raw)
    st.divider()
    _section_preview(res_raw, res_final, k)

    # ---------------------------------------------------------------- gravar
    st.divider()
    default_name = f"{Path(ordered[0].name).stem}_merged.nirs"
    out_name = st.text_input("Nome do arquivo de saída", value=default_name,
                             key=f"{k}_outname")

    if st.button("Concatenar e enviar para a pipeline", type="primary",
                 key=f"{k}_go", width="stretch"):
        out_path = workdir / out_name
        written = core.save_merged(res_final, out_path)

        payload = {
            "path": written["nirs"],
            "result": res_final,
            "block_info": core._block_info_dict(res_final),
            "sidecars": {kk: v for kk, v in written.items() if kk != "nirs"},
        }
        st.session_state[f"{k}_payload"] = payload

        st.success(
            f"{len(ordered)} blocos concatenados — {res_final.n_samples:,} amostras, "
            f"{_fmt_dur(res_final.duration_s)}.".replace(",", ".")
        )

        st.download_button(
            "Baixar .nirs concatenado",
            data=written["nirs"].read_bytes(),
            file_name=out_name,
            mime="application/octet-stream",
            key=f"{k}_dl",
        )

        if on_complete is not None:
            on_complete(payload)

        return payload

    return st.session_state.get(f"{k}_payload")
