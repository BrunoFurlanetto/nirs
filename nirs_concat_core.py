"""
nirs_concat_core.py
===================

Lógica de concatenação de arquivos .nirs (formato Homer) de uma mesma sessão.
Sem dependência de Streamlit — pode ser importado pela pipeline, por scripts
de linha de comando ou pela interface.

Dois modos de junção:

RAW
    Concatena a intensidade bruta sem alterar valores. Só remove os primeiros
    `trim_s` segundos de cada bloco e reconstrói o eixo de tempo.

CORRIGIDO
    Antes de concatenar, reescala cada bloco k por (m_ref / m_k), onde m_k é o
    nível de referência daquele bloco por canal. Isso torna a conversão global
    de Beer-Lambert feita pela pipeline (baseline única no início do arquivo)
    ALGEBRICAMENTE IDÊNTICA a converter cada bloco com a própria baseline:

        od_pipeline = -log(y_k * m_1/m_k) + log(m_1)
                    = -log(y_k) + log(m_k)
                    = od_bloco_k                          (identidade exata)

    O arquivo de saída continua sendo um .nirs válido e a pipeline principal
    não precisa de nenhuma alteração.

Ambos os modos gravam um sidecar com os limites dos blocos, para que a etapa
de detrending possa ser feita bloco a bloco e para que regressores de bloco
possam entrar na matriz de desenho.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.io import loadmat, savemat

__all__ = [
    "NirsFile",
    "MergeResult",
    "JointDiagnostic",
    "load_nirs",
    "check_compatibility",
    "merge_nirs",
    "save_merged",
    "MODE_RAW",
    "MODE_CORRECTED",
    "REF_BASELINE",
    "REF_MEDIAN",
]

MODE_RAW = "raw"
MODE_CORRECTED = "corrected"

REF_BASELINE = "baseline"   # média dos primeiros `baseline_s` do bloco
REF_MEDIAN = "median"       # mediana do bloco inteiro (robusto a artefato inicial)


# --------------------------------------------------------------------------
# Estruturas
# --------------------------------------------------------------------------

@dataclass
class NirsFile:
    """Um arquivo .nirs carregado."""
    name: str
    d: np.ndarray                  # (ns, ncol) intensidade bruta
    t: np.ndarray                  # (ns, 1) tempo
    s: np.ndarray                  # (ns, ncond) estímulos
    aux: np.ndarray                # (ns, naux)
    cond_names: list[str]
    SD: Any                        # struct do MATLAB, repassado sem alteração
    mL: np.ndarray | None
    fs: float
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.d.shape[0])

    @property
    def n_cols(self) -> int:
        return int(self.d.shape[1])

    @property
    def duration_s(self) -> float:
        return float(self.n_samples / self.fs)


@dataclass
class JointDiagnostic:
    """Medida do degrau em uma emenda, por canal."""
    index: int                     # emenda i fica entre bloco i e i+1
    time_s: float                  # instante da emenda no arquivo final
    pct_change: list[float]        # variação % por coluna
    max_abs_pct: float
    n_cols_over_threshold: int
    n_cols: int
    threshold_pct: float

    @property
    def all_channels_shifted(self) -> bool:
        """Todos os canais deslocados aponta para ganho/luz ambiente."""
        return self.n_cols_over_threshold == self.n_cols

    @property
    def some_channels_shifted(self) -> bool:
        """Alguns canais só aponta para acoplamento local (capacete moveu)."""
        return 0 < self.n_cols_over_threshold < self.n_cols

    def verdict(self) -> str:
        if self.n_cols_over_threshold == 0:
            return "sem degrau relevante"
        if self.all_channels_shifted:
            return "todos os canais deslocados — compatível com recalibração de ganho ou luz ambiente"
        return (f"{self.n_cols_over_threshold} de {self.n_cols} canais deslocados — "
                f"compatível com mudança de acoplamento (capacete pode ter se movido)")


@dataclass
class MergeResult:
    """Resultado da concatenação."""
    d: np.ndarray
    t: np.ndarray
    s: np.ndarray
    aux: np.ndarray
    cond_names: list[str]
    SD: Any
    mL: np.ndarray | None
    fs: float
    mode: str
    trim_s: float
    block_starts: list[int]        # índice da 1ª amostra de cada bloco (0-based)
    block_ends: list[int]          # índice da última amostra de cada bloco (0-based, inclusivo)
    block_names: list[str]
    joint_times: list[float]
    diagnostics: list[JointDiagnostic]
    scale_factors: np.ndarray | None   # (nblocos, ncol) — None no modo raw
    warnings: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(self.d.shape[0])

    @property
    def n_cols(self) -> int:
        return int(self.d.shape[1])

    @property
    def duration_s(self) -> float:
        return float(self.n_samples / self.fs)

    def event_counts(self) -> dict[str, int]:
        return {
            name: int((self.s[:, i] > 0).sum())
            for i, name in enumerate(self.cond_names)
        }

    def block_table(self) -> list[dict]:
        return [
            {
                "bloco": i + 1,
                "arquivo": self.block_names[i],
                "amostra_inicial": self.block_starts[i] + 1,   # 1-based p/ MATLAB
                "amostra_final": self.block_ends[i] + 1,
                "inicio_s": round(self.block_starts[i] / self.fs, 2),
                "fim_s": round((self.block_ends[i] + 1) / self.fs, 2),
                "duracao_s": round((self.block_ends[i] - self.block_starts[i] + 1) / self.fs, 2),
            }
            for i in range(len(self.block_starts))
        ]


# --------------------------------------------------------------------------
# Carregamento
# --------------------------------------------------------------------------

def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def _extract_cond_names(raw: dict, n_cond: int) -> list[str]:
    """CondNames vem como array de objetos; normaliza para lista de str."""
    names: list[str] = []
    cn = raw.get("CondNames")
    if cn is not None:
        flat = np.asarray(cn).ravel()
        for item in flat:
            arr = np.asarray(item).ravel()
            names.append(str(arr[0]) if arr.size else "")
    # Preenche nomes faltantes para bater com o número de colunas de s
    while len(names) < n_cond:
        names.append(f"cond{len(names) + 1}")
    return names[:n_cond] if n_cond else names


def load_nirs(path: str | Path, name: str | None = None) -> NirsFile:
    """Carrega um .nirs e valida os campos mínimos."""
    path = Path(path)
    raw = loadmat(str(path), appendmat=False)

    missing = [k for k in ("d", "t", "SD") if k not in raw]
    if missing:
        raise ValueError(
            f"{path.name} não parece um .nirs válido — faltam os campos: {', '.join(missing)}"
        )

    d = _as_2d(raw["d"]).astype(np.float64)
    t = _as_2d(raw["t"]).astype(np.float64)

    if d.shape[0] != t.shape[0]:
        raise ValueError(
            f"{path.name}: d tem {d.shape[0]} amostras mas t tem {t.shape[0]}."
        )
    if d.shape[0] < 2:
        raise ValueError(f"{path.name}: arquivo tem menos de 2 amostras.")

    dt = float(np.median(np.diff(t[:, 0])))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"{path.name}: vetor de tempo inválido.")
    fs = 1.0 / dt

    s = _as_2d(raw["s"]).astype(np.float64) if "s" in raw and np.size(raw["s"]) else \
        np.zeros((d.shape[0], 0))
    aux = _as_2d(raw["aux"]).astype(np.float64) if "aux" in raw and np.size(raw["aux"]) else \
        np.zeros((d.shape[0], 0))

    cond_names = _extract_cond_names(raw, s.shape[1])

    return NirsFile(
        name=name or path.name,
        d=d, t=t, s=s, aux=aux,
        cond_names=cond_names,
        SD=raw["SD"],
        mL=raw.get("mL"),
        fs=fs,
        raw=raw,
    )


# --------------------------------------------------------------------------
# Validação
# --------------------------------------------------------------------------

def _sd_field(SD: Any, field_name: str) -> np.ndarray | None:
    try:
        return np.asarray(SD[0, 0][field_name])
    except Exception:
        return None


def check_compatibility(files: Sequence[NirsFile]) -> tuple[list[str], list[str]]:
    """
    Verifica se os arquivos podem ser concatenados.

    Retorna (erros, avisos). Erros impedem a junção; avisos não.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if len(files) < 2:
        errors.append("São necessários pelo menos 2 arquivos para concatenar.")
        return errors, warnings

    ref = files[0]

    for f in files[1:]:
        if f.n_cols != ref.n_cols:
            errors.append(
                f"{f.name} tem {f.n_cols} colunas, mas {ref.name} tem {ref.n_cols}. "
                "As montagens não são compatíveis."
            )

        if abs(f.fs - ref.fs) > 1e-3:
            errors.append(
                f"{f.name} tem fs={f.fs:.4f} Hz, mas {ref.name} tem fs={ref.fs:.4f} Hz. "
                "Reamostre antes de concatenar."
            )

        lam_f, lam_ref = _sd_field(f.SD, "Lambda"), _sd_field(ref.SD, "Lambda")
        if lam_f is not None and lam_ref is not None and not np.array_equal(lam_f, lam_ref):
            errors.append(
                f"{f.name} usa wavelengths {lam_f.ravel().tolist()}, "
                f"mas {ref.name} usa {lam_ref.ravel().tolist()}."
            )

        ml_f, ml_ref = _sd_field(f.SD, "MeasList"), _sd_field(ref.SD, "MeasList")
        if ml_f is not None and ml_ref is not None and not np.array_equal(ml_f, ml_ref):
            warnings.append(
                f"{f.name} tem MeasList diferente de {ref.name}. "
                "Confira se a montagem é realmente a mesma."
            )

    return errors, warnings


# --------------------------------------------------------------------------
# Concatenação
# --------------------------------------------------------------------------

def _reference_level(d: np.ndarray, fs: float, how: str, baseline_s: float) -> np.ndarray:
    """Nível de referência por coluna, usado para reescalar o bloco."""
    a = np.abs(d)
    if how == REF_MEDIAN:
        ref = np.median(a, axis=0)
    else:
        n = max(1, int(round(baseline_s * fs)))
        n = min(n, a.shape[0])
        ref = a[:n].mean(axis=0)
    # Evita divisão por zero em canal morto
    ref = np.where(np.isfinite(ref) & (ref > 0), ref, 1.0)
    return ref


def _diagnose_joint(prev_tail: np.ndarray,
                    next_head: np.ndarray,
                    index: int,
                    time_s: float,
                    threshold_pct: float) -> JointDiagnostic:
    """Compara o fim de um bloco com o início do seguinte, por canal."""
    a = np.median(np.abs(prev_tail), axis=0)
    b = np.median(np.abs(next_head), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(a > 0, (b - a) / a * 100.0, 0.0)
    pct = np.nan_to_num(pct, nan=0.0, posinf=0.0, neginf=0.0)
    over = int((np.abs(pct) > threshold_pct).sum())
    return JointDiagnostic(
        index=index,
        time_s=round(float(time_s), 2),
        pct_change=[round(float(x), 2) for x in pct],
        max_abs_pct=round(float(np.abs(pct).max()) if pct.size else 0.0, 2),
        n_cols_over_threshold=over,
        n_cols=int(pct.size),
        threshold_pct=threshold_pct,
    )


def merge_nirs(files: Sequence[NirsFile],
               mode: str = MODE_RAW,
               trim_s: float = 5.0,
               reference: str = REF_BASELINE,
               baseline_s: float = 10.0,
               diag_window_s: float = 5.0,
               diag_threshold_pct: float = 5.0) -> MergeResult:
    """
    Concatena os arquivos na ordem em que foram passados.

    Parameters
    ----------
    files : sequência de NirsFile, NA ORDEM CRONOLÓGICA de gravação.
    mode : MODE_RAW ou MODE_CORRECTED.
    trim_s : segundos removidos do início de cada bloco.
    reference : REF_BASELINE ou REF_MEDIAN (usado só no modo corrigido).
    baseline_s : janela de baseline em segundos. Para equivalência exata com a
        pipeline, use o mesmo valor de P.base (padrão do SPM-fNIRS: 10 s).
    diag_window_s : janela usada para medir o degrau em cada emenda.
    diag_threshold_pct : variação a partir da qual um canal conta como deslocado.
    """
    if mode not in (MODE_RAW, MODE_CORRECTED):
        raise ValueError(f"Modo desconhecido: {mode}")

    errors, warnings = check_compatibility(files)
    if errors:
        raise ValueError("Arquivos incompatíveis:\n- " + "\n- ".join(errors))

    ref_file = files[0]
    fs = ref_file.fs
    dt = 1.0 / fs
    trim_n = int(round(trim_s * fs))
    diag_n = max(1, int(round(diag_window_s * fs)))

    # --- União das condições, preservando ordem de primeira aparição ---
    cond_names: list[str] = []
    for f in files:
        for nm in f.cond_names:
            if nm and nm not in cond_names:
                cond_names.append(nm)
    n_cond = len(cond_names)

    # --- Loop de blocos ---
    d_parts, t_parts, s_parts, aux_parts = [], [], [], []
    block_starts, block_ends, block_names = [], [], []
    joint_times: list[float] = []
    diagnostics: list[JointDiagnostic] = []
    scale_rows: list[np.ndarray] = []

    n_aux = max((f.aux.shape[1] for f in files), default=0)
    ref_level: np.ndarray | None = None
    cursor = 0          # índice da próxima amostra no array final
    prev_tail: np.ndarray | None = None

    for k, f in enumerate(files):
        if trim_n >= f.n_samples:
            raise ValueError(
                f"trim_s={trim_s} s remove {f.name} inteiro "
                f"({f.n_samples} amostras a {fs:.1f} Hz)."
            )

        dk = f.d[trim_n:, :]
        sk_raw = f.s[trim_n:, :] if f.s.shape[1] else np.zeros((dk.shape[0], 0))
        auxk = f.aux[trim_n:, :] if f.aux.shape[1] else np.zeros((dk.shape[0], 0))
        nk = dk.shape[0]

        # --- diagnóstico da emenda (sempre no sinal ORIGINAL, antes de escalar) ---
        if prev_tail is not None:
            diagnostics.append(
                _diagnose_joint(prev_tail, dk[:diag_n, :],
                                index=k, time_s=cursor * dt,
                                threshold_pct=diag_threshold_pct)
            )
        prev_tail = dk[-diag_n:, :]

        # --- correção de nível ---
        if mode == MODE_CORRECTED:
            level = _reference_level(dk, fs, reference, baseline_s)
            if ref_level is None:
                ref_level = level
            scale = ref_level / level
            scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
            dk = dk * scale
            scale_rows.append(scale)

        # --- eixo de tempo contínuo ---
        tk = (cursor + np.arange(nk, dtype=np.float64)).reshape(-1, 1) * dt

        # --- estímulos remapeados para a união ---
        sk = np.zeros((nk, n_cond))
        for c, nm in enumerate(f.cond_names):
            if c >= sk_raw.shape[1] or not nm:
                continue
            if nm in cond_names:
                sk[:, cond_names.index(nm)] = sk_raw[:, c]

        # --- aux padronizado ---
        if n_aux:
            ak = np.zeros((nk, n_aux))
            ncopy = min(n_aux, auxk.shape[1])
            if ncopy:
                ak[:, :ncopy] = auxk[:, :ncopy]
        else:
            ak = np.zeros((nk, 0))

        d_parts.append(dk)
        t_parts.append(tk)
        s_parts.append(sk)
        aux_parts.append(ak)

        block_starts.append(cursor)
        block_ends.append(cursor + nk - 1)
        block_names.append(f.name)
        if k < len(files) - 1:
            joint_times.append(round((cursor + nk) * dt, 2))

        cursor += nk

    result = MergeResult(
        d=np.vstack(d_parts),
        t=np.vstack(t_parts),
        s=np.vstack(s_parts) if n_cond else np.zeros((cursor, 0)),
        aux=np.vstack(aux_parts) if n_aux else np.zeros((cursor, 0)),
        cond_names=cond_names,
        SD=ref_file.SD,
        mL=ref_file.mL,
        fs=fs,
        mode=mode,
        trim_s=trim_s,
        block_starts=block_starts,
        block_ends=block_ends,
        block_names=block_names,
        joint_times=joint_times,
        diagnostics=diagnostics,
        scale_factors=np.vstack(scale_rows) if scale_rows else None,
        warnings=warnings,
    )
    return result


# --------------------------------------------------------------------------
# Gravação
# --------------------------------------------------------------------------

def _block_info_dict(res: MergeResult) -> dict:
    """Metadados dos blocos, em 1-based para consumo direto no MATLAB."""
    return {
        "mode": res.mode,
        "fs": res.fs,
        "trim_s": res.trim_s,
        "n_blocks": len(res.block_starts),
        "n_samples": res.n_samples,
        "block_starts": [i + 1 for i in res.block_starts],
        "block_ends": [i + 1 for i in res.block_ends],
        "block_names": res.block_names,
        "joint_times_s": res.joint_times,
        "cond_names": res.cond_names,
    }


def save_merged(res: MergeResult,
                out_path: str | Path,
                write_sidecar: bool = True) -> dict[str, Path]:
    """
    Grava o .nirs concatenado e, opcionalmente, os sidecars de bloco.

    Retorna um dict com os caminhos gravados.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "d": res.d,
        "t": res.t,
        "SD": res.SD,
    }
    if res.s.shape[1]:
        payload["s"] = res.s
        payload["CondNames"] = np.array(
            [np.array([n], dtype=object) for n in res.cond_names], dtype=object
        ).reshape(1, -1)
    if res.aux.shape[1]:
        payload["aux"] = res.aux
    if res.mL is not None:
        payload["mL"] = res.mL

    # Limites dos blocos vão junto no próprio .nirs (campo extra é ignorado
    # por Homer e pelo SPM-fNIRS, então não quebra nada a jusante).
    payload["blockStarts"] = np.array([[i + 1 for i in res.block_starts]], dtype=np.float64)
    payload["blockEnds"] = np.array([[i + 1 for i in res.block_ends]], dtype=np.float64)

    savemat(str(out_path), payload, appendmat=False, do_compression=True)

    written = {"nirs": out_path}

    if write_sidecar:
        info = _block_info_dict(res)

        json_path = out_path.with_name(out_path.stem + "_blocks.json")
        json_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        written["json"] = json_path

        mat_path = out_path.with_name(out_path.stem + "_blocks.mat")
        mat_payload = {
            "fs": float(res.fs),
            "trim_s": float(res.trim_s),
            "mode": res.mode,
            "nBlocks": len(res.block_starts),
            "blockStarts": np.array([[i + 1 for i in res.block_starts]], dtype=np.float64),
            "blockEnds": np.array([[i + 1 for i in res.block_ends]], dtype=np.float64),
            "blockNames": np.array(
                [np.array([n], dtype=object) for n in res.block_names], dtype=object
            ).reshape(1, -1),
            "jointTimes": np.array([res.joint_times], dtype=np.float64),
        }
        if res.scale_factors is not None:
            mat_payload["scaleFactors"] = res.scale_factors
        savemat(str(mat_path), mat_payload, appendmat=False)
        written["mat"] = mat_path

    return written
