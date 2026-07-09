# Editor de Mapeamento no Streamlit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second Streamlit page that ports `editor_mapeamento.py` (tkinter, desktop-only) into the deployed web app, so mapping-CSV editing works on Streamlit Cloud.

**Architecture:** New file `pages/1_Editor_Mapeamento.py`, picked up automatically by Streamlit's native multipage mechanism (any file under `pages/` becomes a sidebar entry alongside `app.py`). Reuses `load_nirs` and `FS` from `pipeline_spm.py` — no changes to `pipeline_spm.py` or `app.py`. `editor_mapeamento.py` (tkinter) is left untouched for local use.

**Tech Stack:** Streamlit (`st.file_uploader`, `st.data_editor`, `st.download_button`), pandas.

## Global Constraints

- Do not modify `app.py` or `pipeline_spm.py` (established project rule — these are hands-off; see memory `feedback_nao_editar_pipeline`).
- One `.nirs` file uploaded at a time (not batch/folder — confirmed with user).
- Output is a download button only — no writing to the server filesystem (Streamlit Cloud container is ephemeral).
- CSV format must stay exactly `arquivo;condicao;ocorrencia;label` (one header line + one row per occurrence), matching what `app.py`'s `mapeamento_file` uploader already parses.
- No automated test suite exists in this repo (pip/streamlit data-pipeline project, no pytest anywhere). Verification is manual: run `streamlit run app.py` locally and drive the page in a browser, per the spec's "Verificação" section. Steps below use manual verification instead of automated tests — this matches existing project conventions, not a deviation from them.

---

### Task 1: Editor de Mapeamento page

**Files:**
- Create: `pages/1_Editor_Mapeamento.py`

**Interfaces:**
- Consumes: `pipeline_spm.load_nirs(path: str) -> dict` (existing, returns dict with key `"events"`: `dict[str, list[float|int]]` — condition name to list of onset samples) and `pipeline_spm.FS` (existing, `float`, sample rate). Both already imported the same way in `app.py`.
- Produces: nothing consumed by other tasks — this is the only task.

This task is one cohesive file built in four verifiable slices (upload+parse, editable table, reset, download) because all four operate on the same in-page state and can't be meaningfully reviewed apart from each other.

- [ ] **Step 1: Create the file with upload + parsing + read-only table**

```python
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
st.dataframe(st.session_state.editor_df.drop(columns=["_default"]), use_container_width=True, hide_index=True)
```

This first slice deliberately renders a plain, read-only `st.dataframe` — the
editable version replaces it in Step 3. It exists so parsing/upload can be
verified in isolation before adding editing behavior on top.

- [ ] **Step 2: Manually verify parsing + upload**

Run: `streamlit run app.py` (from repo root, with `pages/1_Editor_Mapeamento.py` in place — Streamlit auto-discovers it).

In the browser sidebar, click "Editor de Mapeamento". Upload a real `.nirs`
file (e.g. `dp372_pre.nirs`, present in the repo working directory).

Expected: a table appears with columns `tempo`, `evento`, `condicao`,
`ocorrencia`, `label` — one row per event, sorted by onset time ascending,
`label` pre-filled with the condition name stripped of its trailing digits
(e.g. condition `D3` → label `D`).

Upload a non-`.nirs`/corrupted file (or a `.nirs` that doesn't parse):
expected a red `st.error` box with the exception message, no table shown.

- [ ] **Step 3: Replace the read-only table with an editable `st.data_editor`, add reset button**

Replace the final two lines of Step 1 (the `st.subheader` + `st.dataframe`
block) with:

```python
st.subheader(f"Eventos — {st.session_state.editor_filename}")

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Resetar labels"):
        st.session_state.editor_df["label"] = st.session_state.editor_df["_default"]
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
st.session_state.editor_df = edited

st.caption(f"{len(edited)} mapeamento(s) na tabela")
```

The `st.session_state.pop("editor_data_editor", None)` calls (here and in
Step 1) matter: `st.data_editor` keeps its own edit-diff state under its
`key` in `session_state`. Without clearing that key when the backing
dataframe changes (new file loaded, or labels reset), the widget re-applies
stale edits on top of the new data instead of showing it fresh.

- [ ] **Step 4: Manually verify editing + reset**

Run: `streamlit run app.py`, open the editor page, upload a `.nirs` file.

Edit a `label` cell to a custom value, then interact with any other widget
(e.g. resize a column) to trigger a rerun: expected the custom label
persists (not reverted to default).

Select a row and delete it (`num_rows="dynamic"` shows a delete affordance):
expected the row disappears and `st.caption` count decreases by one.

Click "Resetar labels": expected all *remaining* rows' `label` values revert
to their default (condition stripped of trailing digits); the previously
deleted row does NOT come back.

- [ ] **Step 5: Add the CSV download button**

Append after the `st.caption(...)` line from Step 3:

```python
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
```

- [ ] **Step 6: Manually verify the downloaded CSV end-to-end**

Run: `streamlit run app.py`, open the editor page, upload a `.nirs` file,
edit a couple of labels, delete one row, click "Salvar CSV" and save the
downloaded file.

Open the downloaded CSV in a text editor. Expected:
- First line exactly `arquivo;condicao;ocorrencia;label`.
- One line per remaining row, `arquivo` matching the uploaded filename.
- Edited labels reflected; deleted row absent.

Then go to the main page (`app.py`), upload the same `.nirs` file, and
upload the downloaded CSV into the "Mapeamento de condições (CSV)" sidebar
uploader. Expected: no error, and the condition table on the main page
reflects the edited labels (this confirms the format really matches what
`app.py` parses, not just that it looks right).

- [ ] **Step 7: Commit**

```bash
git add pages/1_Editor_Mapeamento.py
git commit -m "feat: add Streamlit page to edit condition mapping in-browser

Ports editor_mapeamento.py (tkinter, local-only) to a Streamlit multipage
app page so mapping CSVs can be reviewed and edited on Streamlit Cloud,
without needing a desktop display server."
```

---

## Self-Review

**Spec coverage:**
- Multipage native (`pages/`) — Task 1 file placement. ✓
- Reuses `load_nirs`/`FS` from `pipeline_spm.py`, no edits to it or `app.py` — Task 1, Global Constraints. ✓
- One `.nirs` at a time — Step 1 `st.file_uploader` (no `accept_multiple_files`). ✓
- `st.data_editor` table, editable `label`, deletable rows — Step 3. ✓
- Reset button reverts labels, not deletions — Step 3 + verified in Step 4. ✓
- Download button, CSV format `arquivo;condicao;ocorrencia;label` — Step 5. ✓
- Error handling via `st.error`, no fake error row — Step 1. ✓
- Manual verification per spec's "Verificação" section, incl. cross-check against `app.py`'s uploader — Step 6. ✓

**Placeholder scan:** No TBD/TODO; every step has complete code or a fully
specified manual check. Clear.

**Type consistency:** `_ler_ocorrencias` returns `list[dict]` with keys
`onset_s: float`, `cond: str`, `idx: int` — used consistently in the
DataFrame construction. `editor_df` columns (`tempo`, `evento`, `condicao`,
`ocorrencia`, `label`, `_default`) are the same set introduced in Step 1 and
consumed in Steps 3 and 5. `st.session_state` keys (`editor_filename`,
`editor_df`, `editor_data_editor`) are used consistently across steps.
