# Editor de Mapeamento no Streamlit — Design

## Problema

`editor_mapeamento.py` é um app tkinter (desktop, GUI local) que permite revisar
os eventos/condições lidos de um `.nirs` e editar o label de saída de cada
ocorrência, salvando um CSV (`arquivo;condicao;ocorrencia;label`) consumido
pelo uploader de mapeamento em `app.py`.

tkinter não roda no Streamlit Cloud (sem display server, sem acesso ao
filesystem do usuário). O app deployado (`app.py`) só cobre a etapa de rodar
o pipeline — a etapa de editar mapeamento continua presa à máquina local.

## Objetivo

Portar a funcionalidade do editor para dentro do app Streamlit já deployado,
como uma segunda página, sem alterar `pipeline_spm.py` ou `app.py`.

## Arquitetura

- Novo arquivo: `pages/1_Editor_Mapeamento.py`
- Streamlit multipage nativo: qualquer arquivo em `pages/` vira uma entrada no
  menu lateral automaticamente, ao lado da página principal (`app.py`).
- Reaproveita `load_nirs` e `FS` de `pipeline_spm.py` (mesma lógica de leitura
  de eventos que `editor_mapeamento.ler_ocorrencias` já usa) — import direto,
  sem duplicar parsing.
- `editor_mapeamento.py` (tkinter) não é removido nem alterado — continua
  disponível pra quem quiser rodar localmente.

## Fluxo

1. Usuário abre a página "Editor Mapeamento" no navegador.
2. `st.file_uploader` — um único `.nirs` por vez (igual ao fluxo principal do
   `app.py`).
3. Arquivo salvo em tempfile (mesmo padrão de `app.py`), `load_nirs` lê os
   eventos, monta lista de ocorrências ordenadas por onset:
   `{onset_s, cond, idx}`.
4. Monta `DataFrame` com colunas:
   - `tempo` (string formatada, ex. `12.34s` ou `1:02.50`) — somente leitura
   - `evento` (ex. `D_3`, `cond_idx`) — somente leitura
   - `label` — editável, default = condição sem sufixo numérico final
     (mesma regra de `_default_label`: `re.sub(r'\d+$', '', cond)`)
5. `st.data_editor(df, disabled=["tempo", "evento"], num_rows="dynamic")` —
   usuário edita labels e apaga linhas indesejadas direto na tabela.
6. Botão "Resetar labels" — reescreve a coluna `label` de volta ao default
   para as linhas que ainda estão na tabela (não restaura linhas apagadas).
7. Botão de download (`st.download_button`) — gera o CSV
   `arquivo;condicao;ocorrencia;label` a partir do estado atual da tabela e
   oferece pra baixar. Usuário sobe esse CSV no uploader de mapeamento da
   página principal.

## Tratamento de erro

Se `load_nirs` falhar (arquivo inválido/corrompido), mostra `st.error(...)`
com a mensagem da exceção e não renderiza a tabela — mais idiomático em
Streamlit do que o hack de linha `[ERRO: ...]` do tkinter.

## Fora de escopo

- Upload de múltiplos arquivos / pasta inteira de uma vez (usuário decidiu
  manter um arquivo por vez, mesmo fluxo de granularidade do `app.py`).
- Persistir o CSV no servidor/repo — o botão de download é a única saída;
  container do Streamlit Cloud é efêmero.
- Qualquer alteração em `pipeline_spm.py` ou `app.py` (fora do escopo,
  intocáveis).

## Verificação

Sem suíte de testes automatizados no projeto atualmente (consistente com o
resto do repo). Verificação manual: `streamlit run app.py` local, abrir a
página do editor, subir um `.nirs` real (ex. `dp372_pre.nirs`), confirmar que
a tabela reflete os eventos esperados, editar/remover linhas, resetar labels,
baixar o CSV e conferir que o formato bate com o que `app.py` espera no
uploader de mapeamento.
