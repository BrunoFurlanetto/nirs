"""
nirs_join_inspector.py
======================

Inspeção visual das emendas de um arquivo concatenado.

Mostra o sinal **sem alteração** e **com correção de nível** lado a lado, em
qualquer estágio do processamento, canal a canal, com as emendas marcadas.

A ideia é dar para olhar e decidir, em vez de aceitar de fé: cada estágio
revela um aspecto diferente da emenda, e o que importa de verdade é como ela
fica na concentração final, depois do detrend.

Uso embutido:

    from nirs_join_inspector import render as render_inspector
    render_inspector(files_ordenados, state_prefix="inspect")

Ou passando um resultado já pronto:

    render_inspector(files, trim_s=5.0)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import streamlit as st

import nirs_concat_core as core
import nirs_stages as stages

__all__ = ["render"]


# --------------------------------------------------------------------------
# Estágios disponíveis
# --------------------------------------------------------------------------

STAGE_INTENSITY = "Intensidade bruta"
STAGE_OD = "Densidade óptica (OD)"
STAGE_HB = "Concentração (sem detrend)"
STAGE_HB_GLOBAL = "Concentração + detrend global"
STAGE_HB_BLOCK = "Concentração + detrend por bloco"

STAGES = [STAGE_INTENSITY, STAGE_OD, STAGE_HB, STAGE_HB_GLOBAL, STAGE_HB_BLOCK]

STAGE_HELP = {
    STAGE_INTENSITY:
        "O sinal como saiu do equipamento. Aqui o degrau de ganho aparece cru — "
        "é o melhor estágio para julgar se os blocos estavam em níveis diferentes.",
    STAGE_OD:
        "Depois do log e da baseline. No modo sem alteração a baseline vem só do "
        "bloco 1, então os blocos seguintes ficam referenciados a algo que não é deles.",
    STAGE_HB:
        "Concentração antes de qualquer filtro. O que sobra na emenda é a diferença "
        "de nível basal entre blocos.",
    STAGE_HB_GLOBAL:
        "Detrend DCT sobre o registro inteiro. Se houver descontinuidade na emenda, "
        "é aqui que o ringing aparece — oscilação em torno do ponto de junção que se "
        "espalha para trechos distantes dela.",
    STAGE_HB_BLOCK:
        "Detrend aplicado dentro de cada bloco separadamente. Cada DCT enxerga só um "
        "trecho contínuo, então não há descontinuidade para gerar ringing.",
}

UNITS = {
    STAGE_INTENSITY: "intensidade (u.a.)",
    STAGE_OD: "OD",
    STAGE_HB: "ΔHb (µM)",
    STAGE_HB_GLOBAL: "ΔHb (µM)",
    STAGE_HB_BLOCK: "ΔHb (µM)",
}


# --------------------------------------------------------------------------
# Cálculo
# --------------------------------------------------------------------------

def _compute_stage(res: core.MergeResult,
                   stage: str,
                   params: stages.OpticalParams,
                   chromo: str,
                   cutoff_s: float) -> np.ndarray:
    """Devolve (ns, nch) para o estágio pedido."""
    ml = np.asarray(res.SD[0, 0]["MeasList"])

    if stage == STAGE_INTENSITY:
        y = stages.split_wavelengths(res.d, ml)
        # mostra o wavelength escolhido — HbO/HbR não existem aqui
        w = 0 if chromo == "HbO" else 1
        return y[:, :, w]

    y = stages.split_wavelengths(res.d, ml)
    od = stages.calc_od(y, res.fs, params.baseline_s)

    if stage == STAGE_OD:
        w = 0 if chromo == "HbO" else 1
        return od[:, :, w]

    hbo, hbr = stages.calc_hb(od, params)
    sig = (hbo if chromo == "HbO" else hbr) * 1000.0   # mM -> µM

    if stage == STAGE_HB:
        return sig
    if stage == STAGE_HB_GLOBAL:
        return stages.detrend_global(sig, res.fs, cutoff_s)
    if stage == STAGE_HB_BLOCK:
        return stages.detrend_per_block(sig, res.block_starts, res.block_ends,
                                        res.fs, cutoff_s)
    raise ValueError(f"Estágio desconhecido: {stage}")


# --------------------------------------------------------------------------
# Gráficos
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Eventos
# --------------------------------------------------------------------------

EVENT_PALETTE = ["#38a169", "#805ad5", "#d69e2e", "#0987a0",
                 "#d53f8c", "#4a5568", "#c05621", "#2f855a"]


def event_times(res: core.MergeResult) -> dict[str, np.ndarray]:
    """Instantes (s) de cada condição no arquivo concatenado."""
    out: dict[str, np.ndarray] = {}
    if res.s.size == 0:
        return out
    for i, name in enumerate(res.cond_names):
        idx = np.where(res.s[:, i] > 0)[0]
        if idx.size:
            out[name] = idx / res.fs
    return out


def events_by_block(res: core.MergeResult) -> list[dict]:
    """Contagem de eventos por bloco e condição."""
    rows = []
    for b, (a, e) in enumerate(zip(res.block_starts, res.block_ends), start=1):
        row = {"bloco": b, "arquivo": res.block_names[b - 1]}
        for i, name in enumerate(res.cond_names):
            if res.s.size == 0:
                continue
            idx = np.where(res.s[:, i] > 0)[0]
            row[name] = int(((idx >= a) & (idx <= e)).sum())
        rows.append(row)
    return rows


def events_near_joint(res: core.MergeResult, pre_s: float) -> list[dict]:
    """
    Eventos cuja janela pré-onset não cabe dentro do próprio bloco.

    Importa porque a subtração de baseline da época só cancela o offset do
    bloco quando a janela pré e a resposta estão no mesmo bloco. Se a janela
    alcança o bloco anterior, a baseline é calculada num nível diferente e a
    época inteira sai deslocada.
    """
    pre_n = int(round(pre_s * res.fs))
    flagged = []
    if res.s.size == 0:
        return flagged

    for i, name in enumerate(res.cond_names):
        for o in np.where(res.s[:, i] > 0)[0]:
            b = next((bi for bi, (a, e) in
                      enumerate(zip(res.block_starts, res.block_ends), start=1)
                      if a <= o <= e), None)
            if b is None:
                continue
            margem = (o - res.block_starts[b - 1]) / res.fs
            if (o - pre_n) < res.block_starts[b - 1]:
                flagged.append({
                    "condição": name,
                    "tempo (s)": round(float(o / res.fs), 1),
                    "bloco": b,
                    "margem (s)": round(float(margem), 1),
                    "precisa (s)": pre_s,
                })
    return flagged


def _event_segments(fig, times_by_cond, y0, y1, row=None, col=None,
                    show_legend=False):
    """
    Desenha os eventos como segmentos verticais.

    Um trace por condição, usando None como separador entre segmentos — bem
    mais rápido que uma vline por evento e ainda gera entrada de legenda.
    """
    import plotly.graph_objects as go

    for i, (name, ts) in enumerate(times_by_cond.items()):
        if len(ts) == 0:
            continue
        xs, ys = [], []
        for x in ts:
            xs.extend([x, x, None])
            ys.extend([y0, y1, None])
        trace = go.Scatter(
            x=xs, y=ys, mode="lines", name=f"evento {name}",
            line=dict(width=1.2, color=EVENT_PALETTE[i % len(EVENT_PALETTE)]),
            legendgroup=f"ev_{name}", showlegend=show_legend,
            hoverinfo="skip",
        )
        if row is None:
            fig.add_trace(trace)
        else:
            fig.add_trace(trace, row=row, col=col)


def _yrange(*arrays) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(a).ravel() for a in arrays if a is not None])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    pad = (hi - lo) * 0.05 or 1.0
    return lo - pad, hi + pad


def _plot_grid(t, sig, res, channels, stage, chromo, key,
               events=None, row_height=300):
    """
    Canais em blocos 2x2, um bloco abaixo do outro.

    Cada grupo de 4 canais vira uma figura independente:

        Canal 1   Canal 2
        Canal 3   Canal 4

        Canal 5   Canal 6
        Canal 7   Canal 8

    Figuras separadas em vez de uma grade única porque assim cada bloco tem o
    próprio eixo e o próprio espaço vertical — a separação entre grupos fica
    visível de fato, não só um espaçamento maior entre linhas.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    groups = [channels[i:i + 4] for i in range(0, len(channels), 4)]

    for gi, group in enumerate(groups):
        ncols = min(2, len(group))
        nrows = int(np.ceil(len(group) / ncols))

        total_h = row_height * nrows
        vspace = min(0.14, 70.0 / total_h) if nrows > 1 else 0.0

        fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=[f"Canal {c + 1}" for c in group],
            shared_xaxes=True,
            vertical_spacing=vspace,
            horizontal_spacing=0.07,
        )

        first_of_all = (gi == 0)

        for i, ch in enumerate(group):
            r, c = i // ncols + 1, i % ncols + 1

            # Eventos primeiro, para ficarem atrás do sinal
            if events:
                y0, y1 = _yrange(sig[:, ch])
                _event_segments(fig, events, y0, y1, row=r, col=c,
                                show_legend=(first_of_all and i == 0))

            fig.add_trace(go.Scatter(
                x=t, y=sig[:, ch], name="sinal",
                line=dict(width=1.2, color="#2b6cb0"),
                legendgroup="sig", showlegend=False,
            ), row=r, col=c)

            for jt in res.joint_times:
                fig.add_vline(x=jt, line_dash="dash", line_color="#e53e3e",
                              line_width=1, row=r, col=c)

        # Só a primeira figura carrega título e legenda; as demais ganham
        # a altura extra de volta em área de gráfico.
        head = 76 if first_of_all else 34
        layout = dict(
            height=total_h + head,
            margin=dict(l=10, r=10, t=head - 12, b=10),
            showlegend=first_of_all,
        )
        if first_of_all:
            layout["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, x=0)
            layout["title"] = dict(text=f"{stage} — {chromo}", x=0.01,
                                   font=dict(size=14))
        fig.update_layout(**layout)

        fig.update_annotations(font_size=12)
        fig.update_xaxes(title_text="Tempo (s)", row=nrows)
        fig.update_yaxes(title_text=UNITS[stage], col=1)

        st.plotly_chart(fig, width="stretch", key=f"{key}_g{gi}")


def _plot_zoom(t, sig, res, channels, joint_idx,
               window_s, stage, chromo, key, events=None):
    """Zoom em torno de uma emenda, com todos os canais escolhidos sobrepostos."""
    import plotly.graph_objects as go

    jt = res.joint_times[joint_idx]
    lo, hi = jt - window_s, jt + window_s
    m = (t >= lo) & (t <= hi)

    fig = go.Figure()
    palette = ["#2b6cb0", "#dd6b20", "#38a169", "#805ad5",
               "#d53f8c", "#319795", "#b7791f", "#e53e3e"]

    # Eventos dentro da janela visível
    if events:
        vis = {n: ts[(ts >= lo) & (ts <= hi)] for n, ts in events.items()}
        vis = {n: ts for n, ts in vis.items() if len(ts)}
        if vis:
            y0, y1 = _yrange(sig[m][:, channels])
            _event_segments(fig, vis, y0, y1, show_legend=True)

    for i, ch in enumerate(channels):
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=t[m], y=sig[m, ch], name=f"Canal {ch + 1}",
            line=dict(width=1.8, color=color),
        ))

    fig.add_vline(x=jt, line_dash="dash", line_color="#e53e3e", line_width=1.5)
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Tempo (s)",
        yaxis_title=UNITS[stage],
        title=dict(text=f"Emenda {joint_idx + 1} em {jt:.1f}s — {stage} ({chromo})",
                   x=0.01, font=dict(size=13)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=9)),
    )
    st.plotly_chart(fig, width="stretch", key=key)


def _step_table(sig, res, channels, stage):
    """Degrau em cada emenda, no sinal tratado."""
    steps = stages.step_sizes(sig, res.block_ends, res.fs)

    rows = []
    for j in range(steps.shape[0]):
        for ch in channels:
            rows.append({
                "emenda": j + 1,
                "canal": ch + 1,
                "degrau": round(float(steps[j, ch]), 5),
            })

    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        f"Degrau em {UNITS[stage]}: mediana dos 5 s após a emenda menos a dos 5 s antes."
    )


# --------------------------------------------------------------------------
# Entrada principal
# --------------------------------------------------------------------------

def render(files: Sequence[core.NirsFile] | None = None,
           trim_s: float = 5.0,
           baseline_s: float = 10.0,
           state_prefix: str = "inspect") -> None:
    """
    Desenha a aba de inspeção.

    Parameters
    ----------
    files : NirsFile já carregados, NA ORDEM cronológica. Se None, mostra um
        uploader próprio.
    trim_s, baseline_s : precisam bater com os usados na concatenação.
    state_prefix : prefixo das chaves de sessão.
    """
    k = state_prefix

    st.subheader("Inspecionar as emendas")
    st.caption(
        "Mostra o sinal já tratado em cada estágio do processamento, com as "
        "emendas e os eventos marcados."
    )

    # ---------------------------------------------------------------- entrada
    if files is None:
        uploaded = st.file_uploader(
            "Arquivos .nirs da mesma sessão (na ordem de gravação)",
            type=["nirs"], accept_multiple_files=True, key=f"{k}_up",
        )
        if not uploaded or len(uploaded) < 2:
            st.info("Envie dois ou mais arquivos .nirs para inspecionar as emendas.")
            return

        import tempfile
        from pathlib import Path
        wd = Path(tempfile.gettempdir()) / f"{k}_in"
        wd.mkdir(parents=True, exist_ok=True)

        loaded = []
        for uf in uploaded:
            p = wd / uf.name
            p.write_bytes(uf.getbuffer())
            try:
                loaded.append(core.load_nirs(p))
            except Exception as exc:
                st.error(f"**{uf.name}** — {exc}")
                return
        files = loaded

    if len(files) < 2:
        st.warning("São necessários pelo menos dois blocos para haver emenda.")
        return

    errors, warns = core.check_compatibility(files)
    for w in warns:
        st.warning(w)
    if errors:
        for e in errors:
            st.error(e)
        return

    # ---------------------------------------------------------------- merges
    try:
        res_raw = core.merge_nirs(files, mode=core.MODE_RAW, trim_s=trim_s)
        res_cor = core.merge_nirs(files, mode=core.MODE_CORRECTED,
                                  trim_s=trim_s, baseline_s=baseline_s)
    except ValueError as exc:
        st.error(str(exc))
        return

    ml = np.asarray(res_raw.SD[0, 0]["MeasList"])
    try:
        nch = stages.split_wavelengths(res_raw.d, ml).shape[1]
    except ValueError as exc:
        st.error(f"Não foi possível separar os wavelengths: {exc}")
        return

    # ---------------------------------------------------------------- controles
    c1, c2 = st.columns([2, 1])
    with c1:
        stage = st.selectbox("Estágio", STAGES, index=4, key=f"{k}_stage")
    with c2:
        chromo = st.radio("Sinal", ["HbO", "HbR"], horizontal=True, key=f"{k}_chromo")

    st.caption(STAGE_HELP[stage])
    if stage == STAGE_INTENSITY:
        st.caption(
            f"Neste estágio HbO/HbR ainda não existem — mostrando o wavelength "
            f"{'752' if chromo == 'HbO' else '840'} nm."
        )

    with st.expander("Parâmetros ópticos e de filtro", expanded=False):
        p1, p2, p3 = st.columns(3)
        with p1:
            dpf = st.number_input("DPF", 1.0, 20.0, 6.61, 0.01, key=f"{k}_dpf")
        with p2:
            dist = st.number_input("Distância (cm)", 0.5, 10.0, 3.5, 0.1, key=f"{k}_dist")
        with p3:
            cutoff = st.number_input("Cutoff do detrend (s)", 8.0, 512.0, 128.0, 8.0,
                                     key=f"{k}_cut")
        st.caption(
            "Os coeficientes de absorção usam a tabela de Cope para 752/840 nm. "
            "Escalas absolutas dependem deles; a comparação entre as duas versões, não."
        )

    params = stages.OpticalParams(
        dpf=(dpf, dpf), distance_cm=dist, baseline_s=baseline_s
    )

    # ---------------------------------------------------------------- cálculo
    try:
        sig = _compute_stage(res_cor, stage, params, chromo, cutoff)
    except ValueError as exc:
        st.error(str(exc))
        return

    channels = st.multiselect(
        "Canais",
        options=list(range(nch)),
        default=list(range(nch)),
        format_func=lambda i: f"Canal {i + 1}",
        key=f"{k}_chs",
    )
    if not channels:
        st.info("Escolha pelo menos um canal.")
        return

    t = res_raw.t[:, 0]

    # ---------------------------------------------------------------- eventos
    all_events = event_times(res_raw)
    events = None

    if all_events:
        e1, e2 = st.columns([3, 1])
        with e1:
            chosen = st.multiselect(
                "Marcar eventos",
                options=list(all_events.keys()),
                default=list(all_events.keys()),
                format_func=lambda n: f"{n} ({len(all_events[n])})",
                key=f"{k}_ev",
            )
        with e2:
            pre_s = st.number_input(
                "Janela pré (s)", 0.0, 120.0, 20.0, 5.0, key=f"{k}_pre",
                help=(
                    "Usada só para checar se algum evento está perto demais do "
                    "início do bloco. Informe a janela pré-onset da sua análise de época."
                ),
            )
        events = {n: all_events[n] for n in chosen} if chosen else None

        with st.expander("Eventos por bloco", expanded=False):
            st.dataframe(events_by_block(res_raw), hide_index=True, width="stretch")
            st.caption(
                "Condições concentradas em blocos diferentes são o caso ruim: o offset "
                "de bloco fica correlacionado com o regressor e enviesa o beta. "
                "Isso é confundimento de desenho e nenhum filtro corrige depois."
            )

        if pre_s > 0:
            flagged = events_near_joint(res_raw, pre_s)
            if flagged:
                st.warning(
                    f"{len(flagged)} evento(s) com margem menor que a janela pré de "
                    f"{pre_s:.0f}s. Nesses, a baseline da época cai no bloco anterior "
                    "e deixa de cancelar o offset — a época inteira sai deslocada. "
                    "Reduza a janela pré ou exclua essas épocas."
                )
                st.dataframe(flagged, hide_index=True, width="stretch")
    else:
        st.caption("Os arquivos não trazem eventos marcados.")

    # ---------------------------------------------------------------- abas
    tab_grid, tab_zoom, tab_table = st.tabs(
        ["Todos os canais", "Zoom na emenda", "Degrau em números"]
    )

    with tab_grid:
        row_h = st.select_slider(
            "Altura das linhas",
            options=[220, 300, 380, 460, 560],
            value=300,
            format_func=lambda v: {220: "compacta", 300: "média", 380: "alta",
                                   460: "muito alta", 560: "máxima"}[v],
            key=f"{k}_rowh",
        )
        _plot_grid(t, sig, res_raw, channels, stage, chromo,
                   key=f"{k}_grid", events=events, row_height=row_h)
        cap = (f"Linhas vermelhas tracejadas: emendas em "
               f"{', '.join(f'{j:.1f}s' for j in res_raw.joint_times)}.")
        if events:
            cap += " Linhas coloridas: eventos, uma cor por condição."
        st.caption(cap)

    with tab_zoom:
        z1, z2 = st.columns([2, 1])
        with z1:
            ji = st.selectbox(
                "Emenda",
                options=list(range(len(res_raw.joint_times))),
                format_func=lambda i: f"Emenda {i + 1} — {res_raw.joint_times[i]:.1f}s",
                key=f"{k}_ji",
            )
        with z2:
            win = st.slider("Janela (± s)", 5, 120, 30, 5, key=f"{k}_win")

        _plot_zoom(t, sig, res_raw, channels, ji, win,
                   stage, chromo, key=f"{k}_zoom", events=events)

        if stage == STAGE_HB_GLOBAL:
            st.info(
                "Compare com o estágio de detrend por bloco. Se aparecer oscilação "
                "em torno da emenda que some no outro estágio, é o ringing do DCT "
                "global — e ele contamina trechos bem distantes da emenda, não só a "
                "vizinhança dela."
            )

    with tab_table:
        _step_table(sig, res_raw, channels, stage)

        if stage in (STAGE_HB, STAGE_HB_GLOBAL, STAGE_HB_BLOCK):
            st.caption(
                "Um degrau residual depois da correção é esperado e correto: cada bloco "
                "passa a ter a concentração referenciada à própria baseline, então "
                "recomeça do próprio zero. O que a correção elimina é o degrau de ganho."
            )
