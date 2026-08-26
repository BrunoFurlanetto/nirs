# Integração: concatenação de arquivos .nirs

Documento para o agente responsável por integrar este módulo ao app principal.

---

## O que é

Módulo que junta vários arquivos `.nirs` de uma mesma sessão num único arquivo,
para casos em que a gravação foi partida em blocos (reinício do software, pausa
entre tarefas).

Pressuposto do desenho: **nenhuma tarefa começa em um arquivo e termina em
outro**. Se isso deixar de valer, a lógica de época precisa ser revista.

---

## Arquivos a copiar

| Arquivo | Papel |
|---|---|
| `nirs_concat_core.py` | Lógica de concatenação. Sem Streamlit. |
| `nirs_stages.py` | Estágios de conversão (OD, Beer-Lambert, detrend) para a visualização. |
| `nirs_concat_ui.py` | Aba de junção. Expõe `render()`. |
| `nirs_join_inspector.py` | Aba de inspeção visual. Expõe `render()`. |
| `nirs_blocos_helper.m` | Funções MATLAB para a etapa de GLM. |

Dependências: `numpy`, `scipy`, `streamlit`, `plotly`.

Os testes (`test_*.py`) não precisam ir para produção, mas vale mantê-los no
repositório — documentam as propriedades que o módulo garante.

---

## Fluxo pedido

```
upload dos .nirs  →  conferir a ordem  →  concatenar  →  pipeline principal
```

A junção **sempre aplica a correção de nível**. Não existe mais escolha de modo
na interface: essa decisão já foi tomada e validada.

O ponto de entrada é `nirs_concat_ui.render()`. Assim que o usuário confirma a
ordem e clica em concatenar, o callback `on_complete` dispara com o arquivo
pronto — é aí que a pipeline principal assume.

### Ligação

```python
from nirs_concat_ui import render as render_merge
from nirs_join_inspector import render as render_inspector

def recebe_merge(payload):
    """Chamado assim que a concatenação termina. Entrega para a pipeline."""
    st.session_state["nirs_path"] = payload["path"]
    st.session_state["nirs_blocks"] = payload["block_info"]
    # payload["result"] já traz d, t, s, SD em memória — não precisa reler do disco

tabs = st.tabs(["Juntar", "Inspecionar", "Raw Data", "Probe", "QC", "HRF", "Convert"])

with tabs[0]:
    render_merge(on_complete=recebe_merge, state_prefix="merge")

with tabs[1]:
    render_inspector(state_prefix="inspect")
```

`state_prefix` isola as chaves de `st.session_state` — use prefixos distintos se
houver mais de uma instância.

Sem callback, o resultado fica em `st.session_state["merge_payload"]`.

### Formato do payload

```python
{
    "path":       Path,          # .nirs concatenado em disco
    "result":     MergeResult,   # .d .t .s .SD .fs .cond_names .block_starts .block_ends
    "block_info": dict,          # limites dos blocos, base 1, prontos para MATLAB
    "sidecars":   dict,          # caminhos do _blocks.json e _blocks.mat
}
```

---

## A correção de nível

Cada bloco é reescalado por `m_ref / m_k`, onde `m_k` é o nível médio daquele
bloco por canal na janela de baseline.

O motivo: a pipeline calcula OD com uma baseline única, tirada do início do
arquivo. Num arquivo concatenado essa baseline pertence só ao bloco 1, e os
blocos seguintes ficariam convertidos contra uma referência que não é deles. A
reescala faz a conta global dar exatamente o mesmo resultado que converter bloco
a bloco:

```
od_pipeline = -log(y_k · m_1/m_k) + log(m_1)
            = -log(y_k) + log(m_k)
            = od_bloco_k
```

É identidade algébrica, não aproximação. `test_core.py` confere numericamente e
o erro fica na ordem de 1e-16.

**Consequência para a integração: a pipeline principal não muda.** O arquivo de
saída é um `.nirs` comum, com os mesmos campos de sempre. A correção acontece
antes, no arquivo.

### Parâmetro que precisa bater

`baseline_s` na UI **tem que ser igual ao `P.base` da pipeline** (padrão do
SPM-fNIRS: 10 s). É esse casamento que sustenta a identidade acima. Se a
pipeline usar outro valor, passe o mesmo em `default_baseline_s`:

```python
render_merge(on_complete=recebe_merge, default_baseline_s=10.0)
```

---

## O que NÃO fazer

**Não reintroduzir a opção de concatenar sem correção.** Foi testada e
descartada — a correção se mostrou claramente melhor. `core.MODE_RAW` continua
existindo no core, mas apenas para uso interno (ver abaixo). Não exponha na UI.

**Não remover o cálculo interno do merge cru.** A UI faz dois merges: o corrigido,
que vai para disco, e um cru que existe só para medir o degrau original. Esse
diagnóstico é o único sinal de que o capacete pode ter se movido entre blocos —
sem ele essa informação se perde silenciosamente. Ver a seção seguinte.

**Não inferir a ordem dos arquivos automaticamente.** Ver "Ordem dos arquivos".

---

## Diagnóstico das emendas: por que ele importa

A UI mede, canal a canal, a diferença entre o fim de um bloco e o começo do
seguinte, **sempre no sinal original antes da correção**. A leitura:

- **Todos os canais deslocados** — compatível com recalibração de ganho no
  restart do software, ou mudança de luz ambiente. É o caso que a correção
  resolve bem.
- **Alguns canais deslocados** — compatível com mudança de acoplamento: o
  capacete se moveu. A correção ainda remove o degrau, mas aí está apagando uma
  diferença física real. Vale checar o QC desses canais antes de seguir.
- **Nenhum canal deslocado** — os blocos já estavam no mesmo nível.

O caso do meio é o que justifica manter o diagnóstico mesmo com a correção
sempre ligada: sem ele, um capacete que reassentou fica indistinguível de um
ganho recalibrado, e a correção esconde a evidência.

---

## Ordem dos arquivos

O eixo de tempo é reconstruído na ordem em que os arquivos são passados, e **não
há como inferir a ordem correta a partir do conteúdo** — cada bloco começa com o
tempo em zero, sem timestamp absoluto.

A UI ordena por nome com ordenação natural (`bloco2` antes de `bloco10`) apenas
como chute inicial, e pede confirmação explícita antes de juntar. Mantenha essa
confirmação.

Se a ordem estiver errada, **o arquivo sai sem erro nenhum**, com os onsets nos
lugares errados. É a falha mais perigosa do módulo justamente por ser silenciosa.

---

## Validações que o módulo já faz

`core.check_compatibility()` retorna `(erros, avisos)`:

**Erros** (impedem a junção): número de colunas diferente, `fs` diferente,
wavelengths diferentes.

**Avisos** (não impedem): `MeasList` diferente entre arquivos.

A UI já apresenta os dois. Se a pipeline chamar o core direto, trate os erros
antes de prosseguir.

---

## Etapa MATLAB: o que ainda precisa ser feito

A correção de nível resolve a conversão. Duas outras coisas não são resolvíveis
dentro do arquivo e ficam a cargo do `nirs_blocos_helper.m`.

### Detrending por bloco

O DCT global ajusta cossenos suaves ao registro inteiro. Cada bloco recomeça a
concentração da própria baseline, então há descontinuidade na emenda — e
cossenos suaves não representam descontinuidade. O ajuste oscila em torno dela
(ringing tipo Gibbs) e a oscilação se espalha para trechos distantes. Isso não
cancela na subtração de baseline da época, porque varia dentro da janela.

Medido em `test_inspector.py` com um degrau puro: o detrend global deixa
oscilação residual no miolo dos blocos, longe das emendas; o detrend por bloco
deixa zero.

```matlab
blocks = blocos_do_nirs('sessao_merged.nirs');   % lê os metadados do próprio .nirs
Y.hbo  = detrend_por_bloco(Y.hbo, blocks, P.fs, P.K.H.cutoff);
Y.hbr  = detrend_por_bloco(Y.hbr, blocks, P.fs, P.K.H.cutoff);
```

Aplique **no lugar** do DCT global, não além dele.

### Regressores de bloco no GLM

Cada bloco tem o próprio nível basal. Sem uma coluna constante por bloco, essa
diferença vira variância não modelada: infla o `ResSS`, o que infla a variância
residual e derruba os t. E se as condições estiverem distribuídas de forma
desigual entre blocos, o offset fica correlacionado com o regressor e o beta sai
enviesado — esse caso é confundimento de desenho e nenhum filtro corrige depois.

```matlab
R = regressores_de_bloco(blocks, P.ns);
SPM.xX.X = [SPM.xX.X(:, 1:end-1), R, SPM.xX.X(:, end)];   % antes da constante
```

O último bloco fica de fora de propósito: a matriz já tem uma constante global,
e incluir todos os blocos deixaria o desenho singular.

### Conferir as janelas pré-onset

A subtração de baseline da época cancela o offset do bloco desde que a janela
pré e a resposta estejam no mesmo bloco. Um onset perto do início de um bloco
quebra isso.

```matlab
checar_eventos_nos_blocos(onsets, blocks, P.fs, 20);   % 20 s de janela pré
```

A aba de inspeção já mostra esse aviso na interface, mas vale repetir no
MATLAB para quem entrar pela pipeline direto.

---

## Metadados de bloco

Os limites dos blocos são gravados em três lugares, todos em base 1:

1. **Dentro do próprio `.nirs`**, como `blockStarts` e `blockEnds`. Campos extras
   são ignorados por Homer e SPM-fNIRS, então não quebram nada a jusante.
2. **`<nome>_blocks.mat`** — sidecar para o MATLAB.
3. **`<nome>_blocks.json`** — sidecar legível.

Preferir a opção 1 quando possível: o arquivo fica autocontido e não há como o
sidecar se perder ou ficar dessincronizado.

---

## Aba de inspeção

`nirs_join_inspector.render()` mostra o sinal **já tratado** em cinco estágios:
intensidade bruta, OD, concentração, concentração + detrend global, concentração
+ detrend por bloco. Abre no último.

A comparação útil é alternar entre os dois últimos: a oscilação que aparece em
torno da emenda no detrend global e some no por bloco é o ringing.

Três visões: grade de canais em blocos 2×2 empilhados; zoom numa emenda com
janela de ±5 a ±120 s; e tabela com o degrau em números.

Eventos aparecem como linha vertical colorida, uma cor por condição, distintas
do vermelho tracejado das emendas.

### Detalhe de implementação a preservar

Cada condição é desenhada como **um único trace com `None` separando os
segmentos**, em vez de uma `add_vline` por evento. Com centenas de eventos a
diferença de desempenho é grande, e a legenda sai de graça. Se for mexer nessa
parte, mantenha a abordagem.

### Coeficientes de absorção

O inspetor usa a tabela de Cope para 752/840 nm, embutida em
`nirs_stages.OpticalParams`. São aproximações — servem para inspeção visual, mas
**não para extrair número absoluto de µM**. Se a inspeção passar a alimentar
algum cálculo quantitativo, troque pelos valores do `acoef_MCope.mat` da
pipeline.

---

## Testes

```bash
python3 test_core.py            # concatenação, identidade algébrica, roundtrip
python3 test_pipeline.py        # arquivo concatenado passando pela conversão
python3 test_inspector.py       # estágios e ringing do DCT
python3 test_ui.py              # aba de junção renderiza
python3 test_inspector_ui.py    # aba de inspeção renderiza
python3 test_inspector_full.py  # inspeção com dados reais em todos os estágios
python3 test_events_ui.py       # eventos e aviso de margem
python3 test_grid_layout.py     # layout em blocos 2×2
```

Os testes fatiam um `.nirs` real em blocos e aplicam ganhos diferentes de
propósito. Se for alterar a lógica de correção ou de detrend, rode
`test_core.py` e `test_inspector.py` — são eles que travam as duas propriedades
centrais.

Os testes esperam um `.nirs` de referência em `/tmp/nirs/dp371/`. Ajuste o
caminho no topo de cada arquivo para o ambiente de destino.
