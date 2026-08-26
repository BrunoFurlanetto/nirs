"""
nirs_stages.py
==============

Reproduz os estágios da pipeline (intensidade → OD → Beer-Lambert → detrend)
para que a inspeção visual mostre a emenda em cada ponto do processamento.

A emenda tem aparência muito diferente em cada estágio:

- Na **intensidade bruta**, o degrau de ganho aparece cru.
- Na **OD**, ele já foi ou não absorvido pela baseline, dependendo do modo.
- Na **concentração**, o que sobra é a diferença de nível basal entre blocos.
- Depois do **detrend global**, aparece o ringing em torno da descontinuidade.
- Depois do **detrend por bloco**, cada trecho é tratado isoladamente.

As convenções seguem `spm_fnirs_calc_od` e `spm_fnirs_calc_hb`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "OpticalParams",
    "split_wavelengths",
    "calc_od",
    "calc_hb",
    "dctmtx",
    "detrend_global",
    "detrend_per_block",
    "step_sizes",
]


# --------------------------------------------------------------------------
# Parâmetros ópticos
# --------------------------------------------------------------------------

@dataclass
class OpticalParams:
    """
    Parâmetros da conversão Beer-Lambert.

    Os valores padrão são os do OctaMon (752/840 nm, DPF 6.61, 35 mm).

    `acoef` são os coeficientes de absorção molar [HbO, HbR] por wavelength,
    interpolados da tabela de Cope. Para bater exatamente com a pipeline, use
    os valores que o `acoef_MCope.mat` fornece para os seus wavelengths — a
    inspeção visual não é sensível a diferenças pequenas aqui, mas números
    absolutos são.
    """
    wavelengths: tuple[float, float] = (752.0, 840.0)
    dpf: tuple[float, float] = (6.61, 6.61)
    distance_cm: float = 3.5
    acoef: tuple[tuple[float, float], tuple[float, float]] = (
        (1.4866, 3.8437),   # 752 nm: HbO, HbR
        (2.5264, 1.7986),   # 840 nm: HbO, HbR
    )
    baseline_s: float = 10.0

    def acoef_array(self) -> np.ndarray:
        return np.asarray(self.acoef, dtype=float)


# --------------------------------------------------------------------------
# Estágios
# --------------------------------------------------------------------------

def split_wavelengths(d: np.ndarray, meas_list: np.ndarray) -> np.ndarray:
    """
    Separa as colunas de `d` por wavelength usando a coluna 4 do MeasList.

    Retorna array (ns, nch, 2).
    """
    ml = np.asarray(meas_list)
    if ml.ndim != 2 or ml.shape[1] < 4:
        raise ValueError("MeasList precisa ter pelo menos 4 colunas.")

    w = ml[:, 3].astype(int)
    i1 = np.where(w == 1)[0]
    i2 = np.where(w == 2)[0]

    if len(i1) != len(i2):
        raise ValueError(
            f"MeasList desbalanceado: {len(i1)} colunas no wavelength 1, "
            f"{len(i2)} no wavelength 2."
        )
    if len(i1) == 0:
        raise ValueError("MeasList não indica nenhuma coluna por wavelength.")

    return np.stack([d[:, i1], d[:, i2]], axis=2)


def calc_od(y: np.ndarray, fs: float, baseline_s: float = 10.0) -> np.ndarray:
    """
    Densidade óptica com baseline única no início do registro.

    Mesma convenção de `spm_fnirs_calc_od`: OD = -log(|y|) + log(mean(|y[base]|)).
    """
    ns, nch, nw = y.shape
    n_base = max(1, min(int(round(baseline_s * fs)), ns))

    a = np.abs(y).reshape(ns, nch * nw, order="F")
    a = np.where(a > 0, a, np.finfo(float).eps)

    od = -np.log(a) + np.log(a[:n_base].mean(axis=0))
    return od.reshape(ns, nch, nw, order="F")


def calc_hb(od: np.ndarray, params: OpticalParams) -> tuple[np.ndarray, np.ndarray]:
    """
    Beer-Lambert modificado. Retorna (HbO, HbR) em mM.

    Mesma convenção de `spm_fnirs_calc_hb`.
    """
    a = np.diag(np.asarray(params.dpf, dtype=float)) @ params.acoef_array()
    det = a[1, 1] * a[0, 0] - a[0, 1] * a[1, 0]
    if abs(det) < 1e-12:
        raise ValueError("Matriz de coeficientes singular — confira acoef e DPF.")

    b = 1.0 / (params.distance_cm * det)
    hbo = b * (a[1, 1] * od[:, :, 0] - a[0, 1] * od[:, :, 1])
    hbr = b * (a[0, 0] * od[:, :, 1] - a[1, 0] * od[:, :, 0])
    return hbo, hbr


def dctmtx(n: int, k: int) -> np.ndarray:
    """Matriz DCT-II, equivalente a `spm_dctmtx(n, k)`."""
    k = max(1, min(int(k), n))
    idx = np.arange(n).reshape(-1, 1)
    order = np.arange(k).reshape(1, -1)
    C = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * idx + 1) * order / (2 * n))
    C[:, 0] = 1.0 / np.sqrt(n)
    return C


def _dct_highpass(seg: np.ndarray, fs: float, cutoff_s: float) -> np.ndarray:
    n = seg.shape[0]
    if n < 3:
        return seg.copy()
    k = max(1, int(2 * n / (cutoff_s * fs)))
    K = dctmtx(n, k + 1)[:, 1:]
    return seg - K @ (K.T @ seg)


def detrend_global(y: np.ndarray, fs: float, cutoff_s: float = 128.0) -> np.ndarray:
    """DCT high-pass sobre o sinal inteiro, como a pipeline faz por padrão."""
    return _dct_highpass(np.asarray(y, dtype=float), fs, cutoff_s)


def detrend_per_block(y: np.ndarray,
                      block_starts: list[int],
                      block_ends: list[int],
                      fs: float,
                      cutoff_s: float = 128.0) -> np.ndarray:
    """
    DCT high-pass aplicado separadamente em cada bloco.

    Índices em base 0, inclusivos no fim (como no MergeResult).
    """
    out = np.array(y, dtype=float, copy=True)
    ns = out.shape[0]
    for a, b in zip(block_starts, block_ends):
        a = max(0, int(a))
        b = min(ns - 1, int(b))
        if b <= a:
            continue
        out[a:b + 1] = _dct_highpass(out[a:b + 1], fs, cutoff_s)
    return out


# --------------------------------------------------------------------------
# Medida do degrau
# --------------------------------------------------------------------------

def step_sizes(sig: np.ndarray,
               block_ends: list[int],
               fs: float,
               window_s: float = 5.0) -> np.ndarray:
    """
    Tamanho do degrau em cada emenda, por canal.

    Compara a mediana da janela imediatamente antes com a imediatamente depois.
    Retorna array (n_emendas, nch) em unidades do próprio sinal.
    """
    sig = np.asarray(sig, dtype=float)
    ns = sig.shape[0]
    w = max(1, int(round(window_s * fs)))

    rows = []
    for end in block_ends[:-1]:
        j = int(end) + 1
        if j <= 0 or j >= ns:
            rows.append(np.zeros(sig.shape[1]))
            continue
        before = np.median(sig[max(0, j - w):j], axis=0)
        after = np.median(sig[j:min(ns, j + w)], axis=0)
        rows.append(after - before)

    return np.vstack(rows) if rows else np.zeros((0, sig.shape[1]))
