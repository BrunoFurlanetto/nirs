"""
pages/1_Editor_Mapeamento.py — Editor de mapeamento de condições fNIRS (Streamlit)

Sobe um arquivo .nirs, lista os eventos individuais lidos do arquivo e
permite editar o label de saída de cada ocorrência. Ao final, gera o CSV
(arquivo;condicao;ocorrencia;label) consumido pelo uploader de mapeamento
da página principal (app.py).

Equivalente web de editor_mapeamento.py (tkinter, uso local).
"""

import os
import re
import tempfile

import pandas as pd
import streamlit as st

from pipeline_spm import load_nirs, FS

st.set_page_config(page_title="Editor de Mapeamento — fNIRS", layout="wide")

st.title("Editor de Mapeamento de Condições")
st.caption(
    "Revise os eventos lidos de um .nirs e edite o label de saída de cada "
    "ocorrência antes de gerar o CSV de mapeamento."
)


def _default_label(cond: str) -> str:
    stripped = re.sub(r"\d+$", "", cond).strip()
    return stripped if stripped else cond


def _ler_ocorrencias(nirs_path: str) -> list[dict]:
    rec = load_nirs(nirs_path)
    rows = []
    for cond, onsets in rec["events"].items():
        for idx, onset in enumerate(sorted(onsets), 1):
            rows.append({
                "onset_s": float(onset) / float(FS),
                "cond": cond,
                "idx": idx,
            })
    return sorted(rows, key=lambda r: r["onset_s"])


def _format_tempo(onset_s: float) -> str:
    mins = int(onset_s // 60)
    secs = onset_s % 60
    return f"{mins}:{secs:05.2f}" if mins > 0 else f"{onset_s:.2f}s"


uploaded = st.file_uploader("Selecione um arquivo .nirs", type=["nirs", "mat"])

if uploaded is None:
    st.info("Selecione um arquivo .nirs para começar.")
    st.stop()

if st.session_state.get("editor_filename") != uploaded.name:
    suffix = os.path.splitext(uploaded.name)[1] or ".nirs"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    try:
        ocorrencias = _ler_ocorrencias(tmp_path)
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        st.session_state.editor_filename = None
        st.stop()

    st.session_state.editor_df = pd.DataFrame([
        {
            "tempo": _format_tempo(o["onset_s"]),
            "evento": f"{o['cond']}_{o['idx']}",
            "condicao": o["cond"],
            "ocorrencia": o["idx"],
            "label": _default_label(o["cond"]),
            "_default": _default_label(o["cond"]),
        }
        for o in ocorrencias
    ])
    st.session_state.editor_filename = uploaded.name
    st.session_state.pop("editor_data_editor", None)

st.subheader(f"Eventos — {st.session_state.editor_filename}")

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Resetar labels"):
        st.session_state.editor_df["label"] = st.session_state.editor_df["_default"]
        st.session_state.editor_df = st.session_state.editor_df.reset_index(drop=True)
        st.session_state.pop("editor_data_editor", None)

edited = st.data_editor(
    st.session_state.editor_df,
    column_config={
        "tempo": st.column_config.TextColumn("Tempo", disabled=True),
        "evento": st.column_config.TextColumn("Evento", disabled=True),
        "condicao": None,
        "ocorrencia": None,
        "label": st.column_config.TextColumn("Label de saída"),
        "_default": None,
    },
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    key="editor_data_editor",
)
edited = edited.reset_index(drop=True)
st.session_state.editor_df = edited

st.caption(f"{len(edited)} mapeamento(s) na tabela")

if len(edited) > 0:
    linhas = ["arquivo;condicao;ocorrencia;label"]
    for _, row in edited.iterrows():
        label = str(row["label"]).strip() or row["condicao"]
        linhas.append(
            f"{st.session_state.editor_filename};{row['condicao']};{row['ocorrencia']};{label}"
        )
    csv_bytes = ("\n".join(linhas) + "\n").encode("utf-8")

    with col_b:
        st.download_button(
            "Salvar CSV",
            data=csv_bytes,
            file_name="mapeamento_condicoes.csv",
            mime="text/csv",
        )
