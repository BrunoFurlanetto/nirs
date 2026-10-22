"""
app.py — Interface Streamlit para a Pipeline SPM fNIRS
=======================================================

Sobe um arquivo .nirs, edita a duração de cada condição numa tabela,
clica em "Rodar pipeline" e visualiza os HRFs.

Requer o arquivo pipeline_spm.py na MESMA PASTA.

Instalação:
    pip install streamlit scipy numpy matplotlib pywavelets plotly

Como rodar:
    streamlit run app.py

Vai abrir no navegador automaticamente (localhost:8501).
"""

import os
import hashlib
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from pipeline_spm import (
    load_nirs,
    compute_channel_distances,
    compute_sci,
    intensity_to_concentration,
    dpf_scholkmann,
    read_dpf_from_nirs,
    baseline_initial_time,
    wavelet_mdl,
    build_design_matrix,
    glm_residuals,
    extract_epochs,
    FS,
)
from nirs_concat_ui import render as render_merge
from nirs_join_inspector import render as render_inspector

st.set_page_config(page_title="Pipeline SPM fNIRS", layout="wide")

st.title("Pipeline SPM fNIRS")
st.caption(
    "Wavelet-MDL → Baseline initial time → GLM (resíduos) — "
    "réplica em Python do pipeline SPM/MATLAB"
)

# ── Estado da sessão ──────────────────────────────────────────────────────────
if "loaded" not in st.session_state:
    st.session_state.loaded = False


def recebe_merge(payload):
    """Callback da aba de junção. Torna o arquivo mesclado disponível para a pipeline."""
    st.session_state["merge_payload"] = payload
    st.session_state["nirs_blocks"] = payload["block_info"]


# ── Sidebar: parâmetros globais ───────────────────────────────────────────────
with st.sidebar:
    st.header("Parâmetros")
    baseline_samples = st.number_input(
        "Baseline — amostras iniciais", min_value=10, max_value=5000,
        value=100, step=10,
        help="Nº de amostras iniciais usadas como linha de base (initial time)",
    )
    wavelet = st.selectbox(
        "Wavelet", ["sym8", "sym6", "db6", "db4", "coif3"], index=0
    )
    wavelet_level = st.number_input(
        "Nível de decomposição", min_value=1, max_value=8, value=4
    )
    st.divider()
    mapeamento_file = st.file_uploader(
        "Mapeamento de condições (CSV)",
        type=["csv"],
        help="CSV com colunas arquivo;condicao;label — renomeia condições por arquivo. "
             "O campo 'arquivo' deve bater com o nome do .nirs carregado.",
    )
    st.divider()
    tmin = st.number_input("Época — início / baseline (s)", value=-20.0, step=1.0)
    post_margin = st.number_input(
        "Margem pós-tarefa (s)", value=0.0, step=1.0,
        help="Segundos após o fim de cada tarefa para capturar a recuperação. "
             "A janela de cada condição = duração + esta margem.",
    )
    bl_start = st.number_input("Baseline da época — início (s)", value=-2.0, step=0.5)
    bl_end = st.number_input("Baseline da época — fim (s)", value=0.0, step=0.5)
    st.divider()
    end_event_letter = st.text_input(
        "Evento de finalização (letra)",
        value="",
        max_chars=1,
        placeholder="Ex: S",
        help="Trunca o sinal na primeira ocorrência desta letra (ex: S → S1). "
             "Vazio = processa o sinal inteiro.",
    ).strip().upper()


# ── Abas principais ───────────────────────────────────────────────────────────
tab_juntar, tab_inspect, tab_pipeline = st.tabs([
    "Juntar arquivos", "Inspecionar emendas", "Pipeline"
])


# ── Aba: Juntar ───────────────────────────────────────────────────────────────
with tab_juntar:
    render_merge(on_complete=recebe_merge, state_prefix="merge", default_baseline_s=10.0)


# ── Aba: Inspecionar ─────────────────────────────────────────────────────────
with tab_inspect:
    render_inspector(state_prefix="inspect")


# ── Aba: Pipeline ─────────────────────────────────────────────────────────────
with tab_pipeline:

    # ── Banner do arquivo mesclado ────────────────────────────────────────────
    merged = st.session_state.get("merge_payload")
    if merged:
        res = merged["result"]
        st.success(
            f"Arquivo mesclado disponível: **{merged['path'].name}** — "
            f"{res.n_samples:,} amostras, {len(res.block_starts)} blocos, "
            f"{res.duration_s:.0f} s.".replace(",", ".")
        )
        if st.button("Carregar arquivo mesclado na pipeline", type="primary",
                     key="load_merged"):
            try:
                tmp_path = str(merged["path"])
                rec = load_nirs(tmp_path)
                st.session_state.rec = rec
                st.session_state.raw_events = dict(rec["events"])
                st.session_state.n_ch = rec["n_ch"]
                st.session_state.filename = merged["path"].name
                st.session_state.tmp_path = tmp_path
                st.session_state.distances = compute_channel_distances(tmp_path)
                st.session_state.dpf_from_file = read_dpf_from_nirs(tmp_path)
                if rec["mode"] == "intensity":
                    st.session_state.sci = compute_sci(rec["intensity"], rec["measlist"])
                else:
                    st.session_state.sci = None
                st.session_state.loaded = True
                st.session_state.events = dict(rec["events"])
            except Exception as e:
                st.error(f"Erro ao carregar arquivo mesclado: {e}")
        st.divider()

    # ── 1. Upload ─────────────────────────────────────────────────────────────
    st.subheader("1. Carregar arquivo")
    uploaded = st.file_uploader("Selecione um arquivo .nirs", type=["nirs", "mat"])

    if uploaded is not None:
        suffix = os.path.splitext(uploaded.name)[1] or ".nirs"
        buf = uploaded.getbuffer()
        file_hash = hashlib.sha256(buf).hexdigest()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(buf)
            tmp_path = tmp.name

        # Recarrega quando o CONTEÚDO muda, não só quando o nome muda.
        # Antes a guarda era `filename != uploaded.name`: ao editar eventos no
        # .nirs e subir de novo com o mesmo nome, load_nirs não era chamado e
        # st.session_state.raw_events ficava com os eventos antigos. O
        # mapeamento era então aplicado em cima de dados velhos, e o evento
        # recém-adicionado nunca aparecia na pipeline.
        if (not st.session_state.loaded) or st.session_state.get("file_hash") != file_hash:
            try:
                rec = load_nirs(tmp_path)
                st.session_state.rec = rec
                st.session_state.raw_events = dict(rec["events"])
                st.session_state.n_ch = rec["n_ch"]
                st.session_state.filename = uploaded.name
                st.session_state.file_hash = file_hash
                st.session_state.tmp_path = tmp_path
                st.session_state.distances = compute_channel_distances(tmp_path)
                st.session_state.dpf_from_file = read_dpf_from_nirs(tmp_path)
                if rec["mode"] == "intensity":
                    st.session_state.sci = compute_sci(rec["intensity"], rec["measlist"])
                else:
                    st.session_state.sci = None
                st.session_state.loaded = True
            except Exception as e:
                st.error(f"Erro ao carregar o arquivo: {e}")
                st.session_state.loaded = False

        # Mapeamento aplicado a cada rerun (garante que tabela atualiza ao trocar CSV)
        if st.session_state.get("loaded"):
            import pandas as _pd
            raw_ev = dict(st.session_state.raw_events)
            mapped_ev = raw_ev

            if mapeamento_file is not None:
                mdf = _pd.read_csv(mapeamento_file, sep=";", dtype=str)
                mapeamento_file.seek(0)
                arquivo_rows = mdf[mdf["arquivo"] == uploaded.name]
                if arquivo_rows.empty:
                    st.warning(
                        f"O CSV de mapeamento não tem nenhuma linha para "
                        f"**{uploaded.name}** — nenhum rótulo foi aplicado. "
                        f"Arquivos no CSV: {', '.join(sorted(mdf['arquivo'].unique()))}"
                    )
                else:
                    if "ocorrencia" in mdf.columns:
                        mapa = {(r["condicao"], r["ocorrencia"]): r["label"]
                                for _, r in arquivo_rows.iterrows()}
                        new_ev: dict = {}
                        sem_linha = []
                        for cond, onsets in raw_ev.items():
                            for i, onset in enumerate(sorted(onsets), 1):
                                label = mapa.get((cond, str(i)))
                                if label is not None:
                                    new_ev.setdefault(label, []).append(onset)
                                else:
                                    sem_linha.append(f"{cond};{i} (t={float(onset):.1f}s)")
                        # O mapeamento é lista de permissão: ocorrência sem linha
                        # no CSV é descartada. Antes isso era silencioso.
                        if sem_linha:
                            st.warning(
                                f"{len(sem_linha)} evento(s) descartado(s) por não terem "
                                f"linha no CSV de mapeamento: {', '.join(sem_linha)}. "
                                "Regere o mapeamento no editor se você editou os eventos do .nirs."
                            )
                        mapped_ev = new_ev
                    else:
                        mapa = {r["condicao"]: r["label"]
                                for _, r in arquivo_rows.iterrows()}
                        mapped_ev = {mapa.get(k, k): v for k, v in raw_ev.items()}

            st.session_state.rec["events"] = mapped_ev
            st.session_state.events = mapped_ev

    if st.session_state.loaded:
        if st.button("🏷️ Editar mapeamento de condições",
                     help="Abre o editor visual para renomear condições deste arquivo"):
            import subprocess, sys as _sys
            editor = os.path.join(os.path.dirname(__file__), "editor_mapeamento.py")
            subprocess.Popen([_sys.executable, editor,
                              "--arquivo", st.session_state.tmp_path,
                              "--nome", st.session_state.filename])

    if st.session_state.loaded:
        rec = st.session_state.rec
        events = st.session_state.events
        n_ch = st.session_state.n_ch
        n_samples = rec["n_samples"]
        filename = st.session_state.filename

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Canais", n_ch)
        c2.metric("Duração", f"{n_samples / FS:.0f} s")
        c3.metric("Condições", len(events))
        c4.metric("Formato", "Intensidade" if rec["mode"] == "intensity" else "Concentração")

        # ── Log: distâncias dos canais ────────────────────────────────────────
        distances = st.session_state.get("distances")
        if distances:
            import pandas as pd
            dmin = min(d for _, d, _ in distances)
            dmax = max(d for _, d, _ in distances)
            orig_unit = distances[0][2]

            log_lines = [f"{lbl}: {dist_mm:.1f} mm" for lbl, dist_mm, _ in distances]
            st.text(
                f"Distâncias fonte-detector (unidade original no arquivo: {orig_unit}):\n"
                + "\n".join(log_lines)
            )

            if dmin < 15:
                st.warning(
                    f"Distância mínima de {dmin:.1f} mm — abaixo de 15 mm sugere "
                    f"erro de escala nas coordenadas (possível troca de unidade no arquivo) "
                    f"ou canal curto. Verifique a SpatialUnit do .nirs."
                )
            elif dmin < 25:
                st.info(
                    f"Distância: {dmin:.1f}–{dmax:.1f} mm — abaixo de 25 mm, "
                    f"penetração cortical reduzida."
                )
            else:
                st.caption(
                    f"Distância: {dmin:.1f}–{dmax:.1f} mm — adequada para fNIRS cerebral."
                )
        else:
            st.caption("Geometria do probe (estrutura SD) não encontrada no arquivo — "
                       "distâncias indisponíveis.")

        import pandas as pd

        # Rótulos dos canais — usar pares S-D das distâncias se disponíveis
        distances = st.session_state.get("distances")
        if distances and len(distances) == n_ch:
            ch_labels = [f"Ch{i+1} ({distances[i][0]})" for i in range(n_ch)]
        else:
            ch_labels = [f"Ch{i+1}" for i in range(n_ch)]

        if rec["mode"] == "intensity":
            st.subheader("2. Beer-Lambert (DPF por comprimento de onda)")

            wavelengths = rec["wavelengths"]
            dpf_from_file = st.session_state.get("dpf_from_file")

            if dpf_from_file:
                st.caption(
                    "O arquivo contém intensidade bruta — será convertido em concentração. "
                    "**DPF encontrado no próprio arquivo** e usado como default (editável)."
                )
                dpf_default = {wl: round(dpf_from_file.get(wl, 6.0), 3) for wl in wavelengths}
                age = None
            else:
                st.caption(
                    "O arquivo contém intensidade bruta — será convertido em concentração. "
                    "O arquivo **não tem DPF armazenado**; o default é calculado pela fórmula "
                    "de Scholkmann & Wolf (2013) para a idade informada (editável)."
                )
                col_age, _ = st.columns([1, 2])
                with col_age:
                    age = st.number_input("Idade do participante (anos)", min_value=1,
                                          max_value=110, value=65, step=1,
                                          help="Usada para calcular o DPF padrão por λ")
                dpf_default = {wl: round(dpf_scholkmann(wl, age), 3) for wl in wavelengths}

            df_dpf = pd.DataFrame({
                "Comprimento de onda (nm)": [int(wl) for wl in wavelengths],
                "DPF": [dpf_default[wl] for wl in wavelengths],
            })
            df_dpf_edit = st.data_editor(
                df_dpf,
                column_config={
                    "Comprimento de onda (nm)": st.column_config.NumberColumn(disabled=True),
                    "DPF": st.column_config.NumberColumn(min_value=1.0, max_value=15.0, step=0.01),
                },
                hide_index=True,
            )
            dpf_values = {
                float(wl): float(dpf)
                for wl, dpf in zip(wavelengths, df_dpf_edit["DPF"])
            }

        # ── Pré-visualização e exclusão de canais ─────────────────────────────
        sec_ch = "3" if rec["mode"] == "intensity" else "2"
        st.subheader(f"{sec_ch}. Inspeção e exclusão de canais")
        st.caption(
            "Visualize o sinal de cada canal (concentração HbO/HbR, sem wavelet) "
            "para decidir quais excluir. Os canais marcados saem de toda a análise."
        )

        if st.button("👁 Pré-visualizar sinal dos canais"):
            with st.spinner("Convertendo e plotando..."):
                if rec["mode"] == "intensity":
                    dist_mm = [d for _, d, _ in st.session_state.distances]
                    prev_oxy, prev_dxy = intensity_to_concentration(
                        rec["intensity"], rec["measlist"], rec["wavelengths"],
                        dist_mm, dpf_values,
                    )
                else:
                    prev_oxy, prev_dxy = rec["oxy"], rec["dxy"]
                st.session_state.prev_oxy = prev_oxy
                st.session_state.prev_dxy = prev_dxy

        if "prev_oxy" in st.session_state:
            prev_oxy = st.session_state.prev_oxy
            prev_dxy = st.session_state.prev_dxy
            t_full = np.arange(prev_oxy.shape[0]) / FS

            calib_n = int(10 * FS)
            vis_oxy = prev_oxy[calib_n:, :]
            vis_dxy = prev_dxy[calib_n:, :]
            t_vis = t_full[calib_n:]
            st.caption("ℹ Os primeiros 10 s (calibração do equipamento) estão ocultos nos gráficos abaixo.")

            shared_scale = st.checkbox("Mesma escala em todos os canais", value=False)
            if shared_scale:
                all_vals = np.concatenate([vis_oxy, vis_dxy])
                _pad = (np.nanmax(all_vals) - np.nanmin(all_vals)) * 0.05
                ylim = (np.nanmin(all_vals) - _pad, np.nanmax(all_vals) + _pad)
            else:
                ylim = None

            # Paleta distinta de HbO (#d62728) e HbR (#1f77b4)
            EVENT_COLORS = ["#38a169", "#805ad5", "#d69e2e", "#0987a0",
                            "#d53f8c", "#c05621", "#2f855a", "#4a5568"]
            cond_list = list(events.items())

            ncols = 4
            rows_ch = [list(range(n_ch))[i:i + ncols] for i in range(0, n_ch, ncols)]
            for row_chs in rows_ch:
                cols = st.columns(ncols)
                for cwidget, ch in zip(cols, row_chs):
                    fig, ax = plt.subplots(figsize=(4, 2.4))
                    ax.set_axisbelow(True)
                    ax.plot(t_vis, vis_oxy[:, ch], color="#d62728", linewidth=0.4, label="HbO")
                    ax.plot(t_vis, vis_dxy[:, ch], color="#1f77b4", linewidth=0.4, label="HbR")
                    if ylim:
                        ax.set_ylim(ylim)
                    xform = ax.get_xaxis_transform()
                    for ci, (cond_name, onsets) in enumerate(cond_list):
                        ev_color = EVENT_COLORS[ci % len(EVENT_COLORS)]
                        label_ch = cond_name[0] if cond_name else "?"
                        for onset in onsets:
                            ax.axvline(onset, color=ev_color, linewidth=1.0,
                                       alpha=0.9, zorder=5)
                            ax.text(onset, 0.98, label_ch, fontsize=5, color=ev_color,
                                    ha="center", va="top", transform=xform, clip_on=True)
                    ax.set_title(ch_labels[ch], fontsize=9)
                    ax.set_xlabel("Tempo (s)", fontsize=7)
                    ax.set_ylabel("µM", fontsize=7)
                    ax.tick_params(labelsize=6)
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=6)
                    cwidget.pyplot(fig)
                    plt.close(fig)

        # Tabela de exclusão
        sci_vals = st.session_state.get("sci")
        SCI_THRESHOLD = 0.7

        def _sci_label(v):
            if v is None or np.isnan(v):
                return "—"
            return "🔴 Baixo" if v < SCI_THRESHOLD else "🟢 OK"

        df_ch = pd.DataFrame({
            "Canal": ch_labels,
            "SCI": [round(float(sci_vals[i]), 3) if sci_vals is not None and i < len(sci_vals) and not np.isnan(sci_vals[i]) else None
                    for i in range(n_ch)],
            "Qualidade": [_sci_label(sci_vals[i] if sci_vals is not None and i < len(sci_vals) else None)
                          for i in range(n_ch)],
            "Excluir": [False] * n_ch,
        })
        df_ch_edit = st.data_editor(
            df_ch,
            column_config={
                "Canal": st.column_config.TextColumn(disabled=True),
                "SCI": st.column_config.NumberColumn(
                    "SCI",
                    disabled=True,
                    format="%.3f",
                    help="Scalp Coupling Index — correlação entre λ1 e λ2 na banda cardíaca (0.5–2.5 Hz). "
                         "Limiar recomendado: ≥ 0.7. N/D para arquivos sem intensidade bruta.",
                ),
                "Qualidade": st.column_config.TextColumn(
                    "Qualidade SCI", disabled=True,
                    help="🔴 Baixo = SCI < 0.7 (acoplamento ruim). 🟢 OK = SCI ≥ 0.7.",
                ),
                "Excluir": st.column_config.CheckboxColumn(
                    "Excluir", help="Marque para remover este canal de toda a análise"
                ),
            },
            hide_index=True,
            use_container_width=True,
        )
        excluded_idx = [i for i, ex in enumerate(df_ch_edit["Excluir"]) if ex]
        keep_idx = [i for i in range(n_ch) if i not in excluded_idx]

        if excluded_idx:
            excluded_names = [ch_labels[i] for i in excluded_idx]
            st.warning(f"Canais excluídos ({len(excluded_idx)}): {', '.join(excluded_names)}. "
                       f"Restam {len(keep_idx)} canais.")
            if not keep_idx:
                st.error("Todos os canais foram excluídos — a análise não pode rodar.")

        # ── Durações por condição ─────────────────────────────────────────────
        sec_num = "4" if rec["mode"] == "intensity" else "3"
        st.subheader(f"{sec_num}. Duração de cada condição")
        st.caption("Edite o valor (em segundos) de cada condição diretamente na tabela.")

        n_trials_por_cond = {k: len(v) for k, v in events.items()}
        df_default = pd.DataFrame({
            "Condição": list(events.keys()),
            "N trials": [n_trials_por_cond[k] for k in events.keys()],
            "Duração (s)": [60.0] * len(events),
            "Início época (s)": [float(tmin)] * len(events),
            "Baseline início (s)": [float(bl_start)] * len(events),
            "Baseline fim (s)": [float(bl_end)] * len(events),
        })

        df_edit = st.data_editor(
            df_default,
            column_config={
                "Condição": st.column_config.TextColumn(disabled=True),
                "N trials": st.column_config.NumberColumn(disabled=True),
                "Duração (s)": st.column_config.NumberColumn(
                    min_value=1.0, max_value=900.0, step=1.0
                ),
                "Início época (s)": st.column_config.NumberColumn(
                    min_value=-300.0, max_value=0.0, step=0.5,
                    help="Início da janela de época relativo ao onset (negativo = antes do evento)"
                ),
                "Baseline início (s)": st.column_config.NumberColumn(
                    min_value=-300.0, max_value=0.0, step=0.5,
                    help="Início da janela de baseline para correção"
                ),
                "Baseline fim (s)": st.column_config.NumberColumn(
                    min_value=-300.0, max_value=300.0, step=0.5,
                    help="Fim da janela de baseline para correção"
                ),
            },
            hide_index=True,
            use_container_width=True,
        )

        durations = dict(zip(df_edit["Condição"], df_edit["Duração (s)"]))
        tmin_per_cond = dict(zip(df_edit["Condição"], df_edit["Início época (s)"]))
        baseline_per_cond = {
            cond: (row["Baseline início (s)"], row["Baseline fim (s)"])
            for cond, row in df_edit.set_index("Condição").iterrows()
        }

        # ── Rodar ──────────────────────────────────────────────────────────────
        sec_run = "5" if rec["mode"] == "intensity" else "4"
        st.subheader(f"{sec_run}. Rodar a pipeline")
        if st.button("▶ Rodar pipeline", type="primary", disabled=(len(keep_idx) == 0)):
            progress = st.progress(0, text="Iniciando...")

            if rec["mode"] == "intensity":
                progress.progress(8, text="Beer-Lambert (intensidade → concentração)...")
                dist_mm = [d for _, d, _ in st.session_state.distances]
                oxy, dxy = intensity_to_concentration(
                    rec["intensity"], rec["measlist"], rec["wavelengths"],
                    dist_mm, dpf_values,
                )
            else:
                oxy, dxy = rec["oxy"], rec["dxy"]

            if excluded_idx:
                oxy = oxy[:, keep_idx]
                dxy = dxy[:, keep_idx]
                kept_labels = [ch_labels[i] for i in keep_idx]
                st.info(f"Análise rodando com {len(keep_idx)} canais: "
                        f"{', '.join(kept_labels)}")
            else:
                kept_labels = ch_labels

            # ── Evento de finalização: janela por trial ────────────────────────
            if end_event_letter:
                from pipeline_spm_end_event import compute_durations_from_end_events
                auto_durs, events, f_onsets = compute_durations_from_end_events(
                    events, end_event_letter
                )
                if auto_durs is None:
                    st.warning(
                        f"Evento de finalização '{end_event_letter}' não encontrado. "
                        "Processando com durações da tabela."
                    )
                else:
                    # Override das durações: onset → próximo evento de finalização
                    durations.update(auto_durs)
                    durations = {k: v for k, v in durations.items() if k in events}
                    tmin_per_cond = {k: v for k, v in tmin_per_cond.items() if k in events}
                    baseline_per_cond = {k: v for k, v in baseline_per_cond.items() if k in events}
                    st.info(
                        f"Evento de finalização '{end_event_letter}': janelas calculadas "
                        f"automaticamente por trial. "
                        + ", ".join(f"**{k}** = {v:.1f}s" for k, v in auto_durs.items())
                    )
                if not events:
                    st.error("Nenhuma condição restante após remover evento de finalização.")
                    st.stop()

            progress.progress(15, text="Baseline initial time...")
            oxy_bl, dxy_bl = baseline_initial_time(oxy, dxy, n_samples=int(baseline_samples))

            progress.progress(35, text="Wavelet-MDL (pode demorar)...")
            oxy_wav = wavelet_mdl(oxy_bl, wavelet=wavelet, level=int(wavelet_level))
            dxy_wav = wavelet_mdl(dxy_bl, wavelet=wavelet, level=int(wavelet_level))

            std_before = np.std(oxy_bl, axis=0)
            std_after = np.std(oxy_wav, axis=0)
            reducao = (1 - std_after / std_before) * 100

            progress.progress(60, text="GLM (matriz de design + OLS)...")
            X = build_design_matrix(n_samples, events, durations, fs=FS)
            oxy_resid = glm_residuals(oxy_wav, X)
            dxy_resid = glm_residuals(dxy_wav, X)

            sig_oxy_for_hrf = oxy_wav
            sig_dxy_for_hrf = dxy_wav

            progress.progress(80, text="Extraindo épocas (sinal pós-wavelet)...")
            epochs_oxy, adj_oxy = extract_epochs(
                sig_oxy_for_hrf, events, tmin=tmin, fs=FS,
                baseline=(bl_start, bl_end),
                durations=durations, post_margin=post_margin,
                tmin_per_cond=tmin_per_cond, baseline_per_cond=baseline_per_cond,
            )
            epochs_dxy, adj_dxy = extract_epochs(
                sig_dxy_for_hrf, events, tmin=tmin, fs=FS,
                baseline=(bl_start, bl_end),
                durations=durations, post_margin=post_margin,
                tmin_per_cond=tmin_per_cond, baseline_per_cond=baseline_per_cond,
            )

            progress.progress(100, text="Concluído!")

            all_adj = {}
            for cond, trials in {**adj_oxy, **adj_dxy}.items():
                all_adj.setdefault(cond, trials)
            for cond, trials in all_adj.items():
                for t in trials:
                    msgs = []
                    if 'tmin_used' in t:
                        msgs.append(
                            f"pré-evento: **{t['tmin_requested']:.1f}s** → **{t['tmin_used']:.1f}s**"
                        )
                    if 'tmax_used' in t:
                        msgs.append(
                            f"pós-evento: **{t['tmax_requested']:.1f}s** → **{t['tmax_used']:.1f}s**"
                        )
                    st.warning(
                        f"⚠ **{cond}** (trial {t['trial']}): janela ajustada por sinal insuficiente — "
                        + ", ".join(msgs),
                        icon=None,
                    )

            st.success("Pipeline concluída.")
            with st.expander("Redução de desvio padrão pelo Wavelet-MDL (por canal)"):
                df_red = pd.DataFrame({
                    "Canal": kept_labels,
                    "Redução std HbO (%)": reducao.round(1),
                })
                st.dataframe(df_red, hide_index=True, use_container_width=True)

            st.subheader("HRF por condição")
            st.caption(
                "Épocas extraídas do sinal **pós-wavelet** (com a resposta preservada). "
                "O GLM é calculado separadamente apenas como diagnóstico — usar os "
                "resíduos do GLM aqui removeria a própria resposta hemodinâmica."
            )
            conditions = [c for c in epochs_oxy if c in epochs_dxy]
            ncols = 3
            rows = [conditions[i:i + ncols] for i in range(0, len(conditions), ncols)]

            hrf_long_rows = []
            hrf_channel_rows = []

            for row in rows:
                cols = st.columns(ncols)
                for col_widget, cond in zip(cols, row):
                    fig, ax = plt.subplots(figsize=(5, 3.2))
                    times = epochs_oxy[cond]["times"]
                    n_trials = epochs_oxy[cond]["n"]

                    hbo = np.mean(epochs_oxy[cond]["mean"], axis=1)
                    hbr = np.mean(epochs_dxy[cond]["mean"], axis=1)
                    hbo_sem = np.mean(epochs_oxy[cond]["sem"], axis=1)
                    hbr_sem = np.mean(epochs_dxy[cond]["sem"], axis=1)

                    oxy_ch = epochs_oxy[cond]["mean"]
                    dxy_ch = epochs_dxy[cond]["mean"]
                    ch_names = [kept_labels[i].split(" ")[0] if i < len(kept_labels)
                                else f"Ch{i+1}" for i in range(oxy_ch.shape[1])]
                    for ti, t_val in enumerate(times):
                        hrf_long_rows.append({
                            "Condicao": cond,
                            "Tempo_s": round(float(t_val), 2),
                            "HbO_uM": float(hbo[ti]),
                            "HbR_uM": float(hbr[ti]),
                            "HbO_SEM": float(hbo_sem[ti]),
                            "HbR_SEM": float(hbr_sem[ti]),
                            "N_trials": n_trials,
                        })
                        row_ch = {"Condicao": cond, "Tempo_s": round(float(t_val), 2)}
                        for ci, ch in enumerate(ch_names):
                            row_ch[f"{ch}_HbO"] = float(oxy_ch[ti, ci])
                            row_ch[f"{ch}_HbR"] = float(dxy_ch[ti, ci])
                        hrf_channel_rows.append(row_ch)

                    if n_trials > 1:
                        ax.fill_between(times, hbo - hbo_sem, hbo + hbo_sem,
                                        color="#d62728", alpha=0.2)
                        ax.fill_between(times, hbr - hbr_sem, hbr + hbr_sem,
                                        color="#1f77b4", alpha=0.2)
                    ax.plot(times, hbo, color="#d62728", label="HbO", linewidth=1.6)
                    ax.plot(times, hbr, color="#1f77b4", label="HbR", linewidth=1.6)
                    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
                    ax.axhline(0, color="gray", alpha=0.3)
                    cond_dur = durations.get(cond, None)
                    if cond_dur is not None:
                        ax.axvspan(0, cond_dur, alpha=0.08, color="green")
                        ax.axvline(cond_dur, color="green", linestyle=":", alpha=0.6)
                    ax.set_title(f"{cond} (n={n_trials}, {durations.get(cond, '?')}s)", fontsize=10)
                    ax.set_xlabel("Tempo (s)", fontsize=8)
                    ax.set_ylabel("µM", fontsize=8)
                    ax.legend(fontsize=7)
                    ax.grid(True, alpha=0.3)
                    col_widget.pyplot(fig)
                    plt.close(fig)

            # ── Download dos dados ────────────────────────────────────────────
            st.subheader("Download")

            import io

            st.markdown("**HRF — valores dos gráficos (CSV)**")
            st.caption("Os mesmos valores mostrados nos gráficos acima, por condição e tempo.")

            df_long = pd.DataFrame(hrf_long_rows)
            df_channel = pd.DataFrame(hrf_channel_rows)

            stem = os.path.splitext(filename)[0]
            cC1, cC2 = st.columns(2)
            with cC1:
                st.download_button(
                    "HRF média entre canais (.csv)",
                    df_long.to_csv(index=False, sep=";").encode("utf-8"),
                    file_name=f"{stem}_hrf_media.csv",
                    mime="text/csv",
                )
            with cC2:
                st.download_button(
                    "HRF por canal (.csv)",
                    df_channel.to_csv(index=False, sep=";").encode("utf-8"),
                    file_name=f"{stem}_hrf_canais.csv",
                    mime="text/csv",
                )

            epoch_rows = []
            for cond in conditions:
                if cond not in epochs_oxy or "epochs" not in epochs_oxy[cond]:
                    continue
                times_ep = epochs_oxy[cond]["times"]
                ep_oxy = epochs_oxy[cond]["epochs"]
                ep_dxy = epochs_dxy[cond]["epochs"]
                ch_names = [kept_labels[i].split(" ")[0] if i < len(kept_labels)
                            else f"Ch{i+1}" for i in range(ep_oxy[0].shape[1])]
                for trial_idx, (eo, ed) in enumerate(zip(ep_oxy, ep_dxy)):
                    for ti, t_val in enumerate(times_ep):
                        row = {"Condicao": cond, "Trial": trial_idx + 1,
                               "Tempo_s": round(float(t_val), 2)}
                        for ci, ch in enumerate(ch_names):
                            row[f"{ch}_HbO"] = float(eo[ti, ci])
                            row[f"{ch}_HbR"] = float(ed[ti, ci])
                        epoch_rows.append(row)

            if epoch_rows:
                df_epochs = pd.DataFrame(epoch_rows)
                st.download_button(
                    "HRF janelas individuais / trials (.csv)",
                    df_epochs.to_csv(index=False, sep=";").encode("utf-8"),
                    file_name=f"{stem}_hrf_trials.csv",
                    mime="text/csv",
                )

            st.divider()

            st.markdown("**Sinal contínuo processado (.npy)**")
            st.caption(
                "Sinal pós-wavelet preserva a resposta (use para HRF). "
                "Resíduos do GLM têm a resposta das condições removida (conectividade)."
            )

            cA, cB = st.columns(2)
            with cA:
                st.markdown("*Pós-wavelet (com resposta)*")
                buf = io.BytesIO(); np.save(buf, oxy_wav)
                st.download_button("HbO pós-wavelet (.npy)", buf.getvalue(),
                    file_name=f"{stem}_oxy_wavelet.npy")
                buf = io.BytesIO(); np.save(buf, dxy_wav)
                st.download_button("HbR pós-wavelet (.npy)", buf.getvalue(),
                    file_name=f"{stem}_dxy_wavelet.npy")
            with cB:
                st.markdown("*Resíduos GLM (sem resposta)*")
                buf = io.BytesIO(); np.save(buf, oxy_resid)
                st.download_button("HbO resíduos (.npy)", buf.getvalue(),
                    file_name=f"{stem}_oxy_residuos.npy")
                buf = io.BytesIO(); np.save(buf, dxy_resid)
                st.download_button("HbR resíduos (.npy)", buf.getvalue(),
                    file_name=f"{stem}_dxy_residuos.npy")

    else:
        st.info("Suba um arquivo .nirs ou carregue um arquivo mesclado para começar.")
