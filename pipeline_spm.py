"""
pipeline_spm.py
---------------
Replica o pipeline SPM de processamento fNIRS descrito no descritivo dp382.

Etapas implementadas (fiel ao SPM):
  1. Carregar .nirs (scipy.io) — dados já em µM exportados pelo OxySoft
  2. Baseline "initial time" — subtrai média do segmento inicial
  3. Wavelet-MDL — remoção de artefatos por Minimum Description Length
  4. GLM com HRF canônica — 11 condições como regressores de confusão
  5. Resíduos GLM → sinal final filtrado
  6. Extração de épocas e geração de gráficos HRF por condição

Uso:
    python pipeline_spm.py arquivo.nirs
    python pipeline_spm.py arquivo.nirs --baseline_samples 100
    python pipeline_spm.py arquivo.nirs --conditions A1 B1 C1 --tmax 60

Dependências:
    pip install scipy numpy matplotlib pywavelets
"""

import sys
import os
import argparse
import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pywt
from scipy.linalg import lstsq
from scipy.special import gamma


# ── Configurações padrão (fiéis ao descritivo dp382) ──────────────────────────
FS = 10.0          # Hz
DPF = 6.61         # fiel ao OxySoft dp382
HRF_DURATION = 32  # segundos (HRF canônica SPM)
HRF_DT = 0.1       # resolução interna (SPM usa 0.01, usamos 0.1 por eficiência)

# Mapeamento condição → duração (segundos)
DEFAULT_DURATIONS = {
    'A1': 60, 'B1': 60, 'C1': 60, 'D1': 60, 'E1': 60,
    'F1': 60, 'H1': 60, 'I1': 60, 'M1': 60, 'P1': 60, 'S1': 60,
}

# Cores para plot
COLORS = {
    'A1': 'red',    'B1': 'blue',   'C1': 'green',  'D1': 'orange',
    'E1': 'purple', 'F1': 'brown',  'H1': 'pink',   'I1': 'cyan',
    'M1': 'olive',  'P1': 'magenta','S1': 'gray',
}


# ── 1. Carregar .nirs ─────────────────────────────────────────────────────────

# Coeficientes de extinção molar — [HbO, HbR] em cm^-1 / (mol/L)
#
# IMPORTANTE (convenção): estes coeficientes são da tabela de Mark Cope (a que o
# NIRS-SPM/OxySoft usa) e são definidos para a densidade óptica em LOGARITMO
# NATURAL (ln) — ver intensity_to_concentration(), que usa np.log (não log10).
# Usar log10 aqui introduz um erro de amplitude de fator ln(10) ≈ 2.303.
#
# Unidade: cm^-1/(mol/L). A tabela de Cope costuma ser citada em cm^-1·mM^-1
# (ex.: 1.2924); aqui está multiplicada por 1000 porque o pipeline resolve a
# concentração em mol/L e converte para µM com ×1e6 (ver final da função).
#
# Os valores abaixo (DEFAULT) foram recuperados do processamento de referência
# do OxySoft (dp382_pre_cog_FINAL.txt): resolvendo OD = ext·L·conc a partir das
# concentrações de referência, reproduzem o FINAL.txt com RMSE ≈ 0. São, na
# prática, a calibração exata do aparelho/OxySoft (OctaMon, 752/840 nm).
#
# Valores tabelados de Cope (literatura), caso queira usar os publicados em vez
# dos recuperados (chegam a ~0.2–0.9 µM de RMSE neste dataset):
#     752: HbO 1349.3, HbR 4066.6   |   840: HbO 2326.0, HbR 1774.0
# Pontos de referência confiáveis do manual (validação): 760 nm (HbO 1486.6 /
# HbR 3843.7) e 830 nm (HbO 2231.4 / HbR 1791.7).
EXTINCTION = {
    752: {'HbO': 1292.4, 'HbR': 3669.9},
    840: {'HbO': 2549.4, 'HbR': 1799.5},
}


def dpf_scholkmann(wavelength_nm, age_years):
    """
    DPF dependente de wavelength e idade — Scholkmann & Wolf (2013).
    Retorna o Differential Pathlength Factor.
    """
    a, b, g = 223.3, 0.05624, 0.8493
    d, e, z = -5.723e-7, 0.001245, -0.9025
    return a + b * (age_years ** g) + d * (wavelength_nm ** 3) \
        + e * (wavelength_nm ** 2) + z * wavelength_nm


def load_nirs(path):
    """
    Carrega arquivo .nirs (formato Homer/OxySoft).

    Detecta automaticamente se 'd' contém:
      - INTENSIDADE bruta (valores ~0-1, precisa de OD + Beer-Lambert)
      - CONCENTRAÇÃO já calculada (oxyData/dxyData em µM)

    Retorna dict com:
      mode          — 'intensity' ou 'concentration'
      intensity     — (amostras × canais×λ) se intensity
      oxy, dxy      — (amostras × canais) se concentration
      wavelengths   — lista de λ
      measlist      — SD.MeasList
      events, n_ch
    """
    data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in data.keys() if not k.startswith('__')]
    print(f"  Chaves encontradas: {keys}")

    result = {'mode': None, 'events': {}, 'wavelengths': None, 'measlist': None}

    # Wavelengths e measlist do SD
    if 'SD' in data:
        SD = data['SD']
        result['wavelengths'] = list(np.atleast_1d(np.array(SD.Lambda)).astype(float))
        result['measlist'] = np.atleast_2d(np.array(SD.MeasList, dtype=int))

    # ── Caso 1: concentração já calculada ──────────────────────────────────────
    if 'oxyData' in data and 'dxyData' in data:
        oxy = np.atleast_2d(data['oxyData'])
        dxy = np.atleast_2d(data['dxyData'])
        if oxy.shape[0] < oxy.shape[1]:
            oxy, dxy = oxy.T, dxy.T
        result['mode'] = 'concentration'
        result['oxy'] = oxy
        result['dxy'] = dxy
        result['n_ch'] = oxy.shape[1]
        n_samples = oxy.shape[0]
        print(f"  Formato: concentração (oxyData/dxyData) — {oxy.shape[1]} canais")

    # ── Caso 2: 'd' presente — detectar intensidade vs concentração ────────────
    elif 'd' in data:
        d = np.atleast_2d(data['d'])
        if d.shape[0] < d.shape[1]:
            d = d.T
        n_samples = d.shape[0]
        n_cols = d.shape[1]

        # Heurística: intensidade óptica tem valores positivos pequenos (~0-1)
        # e n_cols = n_canais × n_wavelengths
        is_intensity = (d.min() >= 0) and (d.max() < 10)

        if is_intensity and result['measlist'] is not None:
            result['mode'] = 'intensity'
            result['intensity'] = d
            n_wl = len(result['wavelengths'])
            result['n_ch'] = n_cols // n_wl
            print(f"  Formato: INTENSIDADE bruta — {result['n_ch']} canais × "
                  f"{n_wl} λ = {n_cols} colunas")
            print(f"  Intensidade range: [{d.min():.4g}, {d.max():.4g}]")
        else:
            # Assumir concentração intercalada
            n_ch = n_cols // 2
            result['mode'] = 'concentration'
            result['oxy'] = d[:, :n_ch]
            result['dxy'] = d[:, n_ch:]
            result['n_ch'] = n_ch
            print(f"  Formato: concentração ('d' dividido) — {n_ch} canais")
    else:
        raise ValueError(f"Formato não reconhecido. Chaves: {keys}")

    print(f"  {n_samples} amostras @ {FS} Hz = {n_samples/FS:.1f}s")

    # ── Eventos ────────────────────────────────────────────────────────────────
    events = {}
    if 's' in data:
        s = np.atleast_2d(data['s'])
        if s.shape[0] < s.shape[1]:
            s = s.T
        for col in range(s.shape[1]):
            onset_samples = np.where(s[:, col] > 0)[0]
            if len(onset_samples) > 0:
                events[f'cond_{col+1}'] = onset_samples / FS

    if 'CondNames' in data:
        cond_names = data['CondNames']
        if hasattr(cond_names, '__iter__') and not isinstance(cond_names, str):
            named = {}
            for i, name in enumerate(cond_names):
                old = f'cond_{i+1}'
                if old in events:
                    named[str(name).strip()] = events[old]
            if named:
                events = named
    print(f"  Condições: {list(events.keys())}")

    result['events'] = events
    result['n_samples'] = n_samples
    return result


def intensity_to_concentration(intensity, measlist, wavelengths, distances_mm,
                                dpf_values):
    """
    Converte intensidade óptica bruta em concentração de HbO/HbR.

    Etapas:
      1. Intensidade → Densidade óptica: OD = -log10(I / I_média)
      2. Beer-Lambert modificada → ΔHbO, ΔHbR

    Parâmetros:
      intensity    — (amostras × (canais×λ))
      measlist     — SD.MeasList (linhas: [src, det, 1, wl_index])
      wavelengths  — lista de λ (ex: [752, 840])
      distances_mm — distância de cada canal em mm
      dpf_values   — dict {wavelength: dpf}

    Retorna oxy, dxy (amostras × canais) em µM.
    """
    n_samples = intensity.shape[0]
    n_wl = len(wavelengths)
    n_ch = intensity.shape[1] // n_wl

    # 1. Densidade óptica por coluna
    # OD em LOGARITMO NATURAL (ln). A lei de Beer-Lambert modificada de
    # Cope & Delpy (1988) e os coeficientes de extinção de Cope usados acima
    # são definidos para ln. Usar log10 introduz erro de amplitude de ln(10).
    mean_int = np.mean(intensity, axis=0, keepdims=True)
    mean_int[mean_int <= 0] = 1e-12
    od = -np.log(intensity / mean_int)

    # 2. Beer-Lambert por canal
    # Agrupar colunas por canal e wavelength via measlist
    # measlist[i] = [src, det, 1, wl_index]; coluna i corresponde a essa medição
    oxy = np.zeros((n_samples, n_ch))
    dxy = np.zeros((n_samples, n_ch))

    # Mapear: para cada canal (par src-det), achar as colunas de cada λ
    pairs = {}
    for col, row in enumerate(measlist):
        src, det, _, wl_idx = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        key = (src, det)
        pairs.setdefault(key, {})[wl_idx] = col

    for ch_idx, (key, wl_cols) in enumerate(pairs.items()):
        if ch_idx >= n_ch:
            break
        d_mm = distances_mm[ch_idx] if ch_idx < len(distances_mm) else 35.0
        # Extinção em cm^-1 → comprimento de caminho em cm (L = d_cm × DPF).
        # Esta conversão estava correta; o erro de amplitude vinha do log10 e
        # dos coeficientes de extinção, não da unidade de distância.
        d_cm = d_mm / 10.0

        # Montar matriz de extinção 2x2 [λ1,λ2] × [HbO,HbR]
        ext_matrix = np.zeros((n_wl, 2))
        od_ch = np.zeros((n_samples, n_wl))
        L = np.zeros(n_wl)

        for wi, wl in enumerate(wavelengths):
            wl_int = int(round(wl))
            # Coeficiente de extinção mais próximo disponível
            wl_key = min(EXTINCTION.keys(), key=lambda k: abs(k - wl_int))
            ext_matrix[wi, 0] = EXTINCTION[wl_key]['HbO']
            ext_matrix[wi, 1] = EXTINCTION[wl_key]['HbR']
            # Comprimento de caminho = distância × DPF
            L[wi] = d_cm * dpf_values[wl]
            # Coluna de OD desse λ (wl_idx é 1-based no measlist)
            col = wl_cols.get(wi + 1)
            if col is not None:
                od_ch[:, wi] = od[:, col]

        # Resolver sistema: OD = ext × L × conc  →  conc = (ext×L)^-1 × OD
        A = ext_matrix * L[:, None]  # (n_wl × 2)
        try:
            A_inv = np.linalg.pinv(A)
        except np.linalg.LinAlgError:
            continue
        conc = od_ch @ A_inv.T  # (n_samples × 2), em mol/L

        oxy[:, ch_idx] = conc[:, 0] * 1e6  # → µM
        dxy[:, ch_idx] = conc[:, 1] * 1e6

    return oxy, dxy


def read_dpf_from_nirs(path):
    """
    Tenta extrair o DPF armazenado no arquivo .nirs, procurando em campos
    comuns onde diferentes exportadores costumam guardá-lo.

    Retorna dict {wavelength: dpf} se encontrar, ou None caso contrário.

    Locais verificados:
      - SD.DPF / SD.ppf / SD.PPF / SD.pathlength
      - chave top-level 'DPF' / 'ppf' / 'dpf'
    """
    data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)

    wavelengths = None
    if 'SD' in data:
        SD = data['SD']
        try:
            wavelengths = list(np.atleast_1d(np.array(SD.Lambda)).astype(float))
        except AttributeError:
            pass

        # Procurar atributos de DPF na estrutura SD
        for attr in dir(SD):
            low = attr.lower()
            if low in ('dpf', 'ppf', 'pathlength', 'pathlengthfactor'):
                val = getattr(SD, attr)
                arr = np.atleast_1d(np.array(val, dtype=float))
                if wavelengths and len(arr) == len(wavelengths):
                    return {wl: float(v) for wl, v in zip(wavelengths, arr)}
                elif len(arr) == 1 and wavelengths:
                    return {wl: float(arr[0]) for wl in wavelengths}

    # Procurar chave top-level
    for key in ('DPF', 'dpf', 'ppf', 'PPF'):
        if key in data:
            arr = np.atleast_1d(np.array(data[key], dtype=float))
            if wavelengths and len(arr) == len(wavelengths):
                return {wl: float(v) for wl, v in zip(wavelengths, arr)}
            elif len(arr) == 1 and wavelengths:
                return {wl: float(arr[0]) for wl in wavelengths}

    return None


def compute_channel_distances(path):
    """
    Lê a estrutura SD do arquivo .nirs e calcula a distância fonte-detector
    de cada canal. Retorna lista de (rotulo, distancia_mm) ou None se não houver SD.

    IMPORTANTE: a distância retornada é SEMPRE em mm, convertida a partir da
    SpatialUnit declarada no arquivo (cm, mm ou m). Isso evita confusão de
    unidade ao comparar com limiares.

    O formato Homer/OxySoft guarda:
      SD.SrcPos     — posições das fontes (n_src × 3)
      SD.DetPos     — posições dos detectores (n_det × 3)
      SD.MeasList   — lista de medições (cada linha: [src, det, 1, wavelength])
      SD.SpatialUnit — unidade espacial ('cm', 'mm', 'm')
    """
    data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    if 'SD' not in data:
        return None

    SD = data['SD']
    try:
        src = np.atleast_2d(np.array(SD.SrcPos, dtype=float))
        det = np.atleast_2d(np.array(SD.DetPos, dtype=float))
        meas = np.atleast_2d(np.array(SD.MeasList, dtype=float))
    except AttributeError:
        return None

    # Unidade espacial declarada — converter tudo para mm
    unit = getattr(SD, 'SpatialUnit', 'mm')
    if not isinstance(unit, str):
        unit = 'mm'
    unit = unit.strip().lower()

    to_mm = {'mm': 1.0, 'cm': 10.0, 'm': 1000.0}.get(unit, 1.0)

    results = []
    seen = set()
    for row in meas:
        si, di = int(row[0]) - 1, int(row[1]) - 1
        # Cada par fonte-detector aparece 2x (um por wavelength); contar uma vez
        if (si, di) in seen:
            continue
        seen.add((si, di))
        if si < len(src) and di < len(det):
            dist_native = float(np.linalg.norm(src[si] - det[di]))
            dist_mm = dist_native * to_mm
            results.append((f"S{si+1}-D{di+1}", dist_mm, unit))

    return results


# ── 2. Baseline "initial time" ────────────────────────────────────────────────

def baseline_initial_time(oxy, dxy, n_samples=100):
    """
    Subtrai a média dos primeiros n_samples de cada canal.
    Fiel ao SPM 'initial time' baseline correction.
    """
    bl_oxy = np.mean(oxy[:n_samples, :], axis=0, keepdims=True)
    bl_dxy = np.mean(dxy[:n_samples, :], axis=0, keepdims=True)
    return oxy - bl_oxy, dxy - bl_dxy


# ── 3. Wavelet-MDL ────────────────────────────────────────────────────────────

def _mdl_threshold(coeffs):
    """
    Critério MDL (Minimum Description Length) para selecionar threshold.
    Implementação baseada em Molavi & Dumont (2012) adaptada para MDL.
    """
    n = len(coeffs)
    abs_c = np.abs(coeffs)
    sorted_c = np.sort(abs_c)[::-1]

    mdl = np.inf
    best_k = 0

    for k in range(1, n):
        # Componente de sinal (k maiores coeficientes)
        signal_power = np.sum(sorted_c[:k] ** 2)
        # Componente de ruído (n-k menores)
        noise_var = np.sum(sorted_c[k:] ** 2) / max(n - k, 1)
        if noise_var <= 0:
            continue
        # Custo MDL
        cost = (n - k) * np.log(noise_var) + k * np.log(signal_power / k + 1e-10) + k * np.log(n)
        if cost < mdl:
            mdl = cost
            best_k = k

    if best_k == 0 or best_k >= n:
        return sorted_c[0]
    return sorted_c[best_k]


def wavelet_mdl(signal, wavelet='sym8', level=4):
    """
    Remoção de artefatos por Wavelet-MDL.
    Fiel ao método usado no SPM fNIRS (Brigadoi et al. 2014).
    """
    n_samples, n_ch = signal.shape
    corrected = np.zeros_like(signal)

    for ch in range(n_ch):
        x = signal[:, ch].copy()

        # Pad para potência de 2
        n_pad = 2 ** int(np.ceil(np.log2(len(x))))
        x_pad = np.pad(x, (0, n_pad - len(x)), mode='reflect')

        # Decomposição wavelet
        coeffs = pywt.wavedec(x_pad, wavelet, level=level)

        # Aplicar threshold MDL em cada escala (exceto aproximação)
        coeffs_thresh = [coeffs[0]]  # mantém aproximação intacta
        for detail in coeffs[1:]:
            thr = _mdl_threshold(detail)
            # Soft thresholding
            detail_thresh = pywt.threshold(detail, thr, mode='soft')
            coeffs_thresh.append(detail_thresh)

        # Reconstrução
        x_rec = pywt.waverec(coeffs_thresh, wavelet)[:len(x)]
        corrected[:, ch] = x_rec

    return corrected


# ── 4. HRF Canônica SPM ───────────────────────────────────────────────────────

def spm_hrf(dt=0.1, duration=32):
    """
    HRF canônica do SPM — dupla função gama.
    Fiel aos parâmetros padrão do SPM8/SPM12.
    """
    t = np.arange(0, duration, dt)

    # Parâmetros SPM padrão
    p1 = 6.0   # pico 1 (s)
    p2 = 16.0  # pico 2 (s)
    d1 = 1.0   # dispersão 1
    d2 = 1.0   # dispersão 2
    ratio = 6.0  # proporção peak/undershoot

    def gamma_pdf(t, shape, scale):
        return (t ** (shape - 1) * np.exp(-t / scale) /
                (scale ** shape * gamma(shape)))

    h = (gamma_pdf(t, p1 / d1, d1) -
         gamma_pdf(t, p2 / d2, d2) / ratio)

    h = h / np.max(np.abs(h))  # normalizar
    return h


def build_design_matrix(n_samples, events, durations, fs=10.0, hrf_dt=0.1):
    """
    Monta matriz de design GLM com HRF canônica convoluída.
    Fiel ao SPM: 11 regressores de condição + 1 constante.
    """
    hrf = spm_hrf(dt=hrf_dt, duration=HRF_DURATION)
    n_conds = len(events)
    X = np.zeros((n_samples, n_conds + 1))

    for col, (cond_name, onsets) in enumerate(events.items()):
        duration = durations.get(cond_name, 60.0)

        # Criar vetor de estímulo em alta resolução
        n_hires = int(n_samples / fs / hrf_dt)
        stim = np.zeros(n_hires)

        for onset in onsets:
            i_start = int(onset / hrf_dt)
            i_end = int((onset + duration) / hrf_dt)
            i_start = max(0, min(i_start, n_hires))
            i_end = max(0, min(i_end, n_hires))
            stim[i_start:i_end] = 1.0

        # Convolução com HRF
        convolved = np.convolve(stim, hrf)[:n_hires]

        # Downsample para fs
        step = int(1.0 / (fs * hrf_dt))
        downsampled = convolved[::step]
        n = min(len(downsampled), n_samples)
        X[:n, col] = downsampled[:n]

    # Constante de sessão
    X[:, -1] = 1.0

    return X


# ── 5. GLM OLS e resíduos ─────────────────────────────────────────────────────

def glm_residuals(signal, X):
    """
    Ajusta GLM por OLS e retorna resíduos.
    Fiel ao SPM: resíduos = sinal - X @ beta.
    """
    n_samples, n_ch = signal.shape
    residuals = np.zeros_like(signal)

    for ch in range(n_ch):
        y = signal[:, ch]
        beta, _, _, _ = lstsq(X, y)
        y_hat = X @ beta
        residuals[:, ch] = y - y_hat

    return residuals


# ── 6. Extração de épocas ─────────────────────────────────────────────────────

def extract_epochs(signal, events, tmin=-20, tmax=60, fs=10.0, baseline=(-2, 0),
                   durations=None, post_margin=15.0):
    """
    Extrai e faz média das épocas por condição com baseline correction.

    Parâmetros:
      tmax        — fim da janela (s). Usado se durations=None ou condição ausente.
      durations   — dict {condição: duração_s}. Se fornecido, a janela de cada
                    condição vai de tmin até (duração + post_margin), respeitando
                    o tempo individual de cada tarefa.
      post_margin — segundos adicionais após o fim da tarefa, para capturar a
                    recuperação hemodinâmica (padrão 15s).

    Retorna:
      (epochs_data, adjustments)
      adjustments — dict {cond: [{trial, tmin_requested, tmin_used,
                                   tmax_requested, tmax_used}, ...]}
                    presente apenas para trials com janela recortada.
    """
    epochs_data = {}
    adjustments = {}
    n_sig = len(signal)

    for cond_name, onsets in events.items():
        # Determinar tmax desta condição
        if durations is not None and cond_name in durations:
            cond_tmax = float(durations[cond_name]) + post_margin
        else:
            cond_tmax = tmax

        n_pre = int(round(abs(tmin) * fs))
        n_post = int(round(cond_tmax * fs))
        n_samples_epoch = n_pre + n_post
        # Grid de tempo a partir da taxa de amostragem (passo fixo = 1/fs),
        # evita decimais imprecisos do linspace (ex: -19.399 em vez de -19.4)
        times = tmin + np.arange(n_samples_epoch) / fs

        epochs = []
        cond_adj = []

        for trial_idx, onset in enumerate(onsets):
            i_onset = int(onset * fs)
            start = i_onset - n_pre
            end = i_onset + n_post

            pad_before = max(0, -start)
            pad_after = max(0, end - n_sig)
            clipped_start = max(0, start)
            clipped_end = min(n_sig, end)

            if clipped_start >= clipped_end:
                continue

            epoch = signal[clipped_start:clipped_end, :]

            adj = {}
            if pad_before > 0:
                epoch = np.pad(epoch, ((pad_before, 0), (0, 0)), mode='constant')
                adj['tmin_requested'] = tmin
                adj['tmin_used'] = round(-(i_onset / fs), 3)
            if pad_after > 0:
                epoch = np.pad(epoch, ((0, pad_after), (0, 0)), mode='constant')
                adj['tmax_requested'] = cond_tmax
                adj['tmax_used'] = round((n_sig - i_onset) / fs, 3)

            if adj:
                adj['trial'] = trial_idx + 1
                cond_adj.append(adj)

            # Baseline correction
            bl_s = int((baseline[0] - tmin) * fs)
            bl_e = int((baseline[1] - tmin) * fs)
            if bl_e > bl_s:
                bl_mean = np.mean(epoch[bl_s:bl_e, :], axis=0, keepdims=True)
                epoch = epoch - bl_mean

            epochs.append(epoch)

        if cond_adj:
            adjustments[cond_name] = cond_adj

        if epochs:
            epochs_data[cond_name] = {
                'mean': np.mean(epochs, axis=0),
                'sem': np.std(epochs, axis=0) / np.sqrt(len(epochs)) if len(epochs) > 1 else np.zeros_like(epochs[0]),
                'n': len(epochs),
                'times': times,
                'epochs': epochs,  # lista de arrays (tempo × canais), um por trial
            }

    return epochs_data, adjustments


# ── 7. Plots ──────────────────────────────────────────────────────────────────

def plot_hrfs(epochs_oxy, epochs_dxy, output_path, participant_id):
    """
    Gera grid de HRFs por condição (HbO e HbR, média entre canais).
    """
    conditions = [c for c in epochs_oxy if c in epochs_dxy]
    if not conditions:
        print("  Nenhuma condição para plotar.")
        return

    n = len(conditions)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))
    axes = axes.flatten() if n > 1 else [axes]

    for ax_idx, cond in enumerate(conditions):
        ax = axes[ax_idx]
        times = epochs_oxy[cond]['times']
        n_trials = epochs_oxy[cond]['n']

        hbo_mean = np.mean(epochs_oxy[cond]['mean'], axis=1)
        hbr_mean = np.mean(epochs_dxy[cond]['mean'], axis=1)
        hbo_sem = np.mean(epochs_oxy[cond]['sem'], axis=1)
        hbr_sem = np.mean(epochs_dxy[cond]['sem'], axis=1)

        if n_trials > 1:
            ax.fill_between(times, hbo_mean - hbo_sem, hbo_mean + hbo_sem,
                            color='#d62728', alpha=0.2)
            ax.fill_between(times, hbr_mean - hbr_sem, hbr_mean + hbr_sem,
                            color='#1f77b4', alpha=0.2)

        ax.plot(times, hbo_mean, color='#d62728', label='HbO', linewidth=1.8)
        ax.plot(times, hbr_mean, color='#1f77b4', label='HbR', linewidth=1.8)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax.set_title(f'{cond}  (n={n_trials})')
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('µM')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(conditions):]:
        ax.set_visible(False)

    plt.suptitle(f'{participant_id} — HRF (Pipeline SPM replicada em Python)\n'
                 f'Wavelet-MDL → Baseline initial time → GLM resíduos', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  HRFs salvas em: {output_path}")


def plot_signal_complete(oxy_final, dxy_final, events, output_path, participant_id):
    """
    Plota sinal completo após pipeline com eventos marcados.
    """
    n_samples, n_ch = oxy_final.shape
    t = np.arange(n_samples) / FS

    fig, axes = plt.subplots(2, 1, figsize=(18, 8))
    colors_list = plt.cm.tab10(np.linspace(0, 1, n_ch))

    for ch in range(n_ch):
        axes[0].plot(t, oxy_final[:, ch], alpha=0.5, linewidth=0.5,
                     color=colors_list[ch], label=f'Ch{ch+1}')
        axes[1].plot(t, dxy_final[:, ch], alpha=0.5, linewidth=0.5,
                     color=colors_list[ch])

    axes[0].set_title('HbO — sinal final (resíduos GLM)')
    axes[0].set_ylabel('µM')
    axes[0].legend(fontsize=7, loc='upper right', ncol=4)

    axes[1].set_title('HbR — sinal final (resíduos GLM)')
    axes[1].set_ylabel('µM')
    axes[1].set_xlabel('Tempo (s)')

    # Marcar eventos
    ev_colors = list(COLORS.values())
    for i, (cond, onsets) in enumerate(events.items()):
        c = COLORS.get(cond, ev_colors[i % len(ev_colors)])
        for onset in onsets:
            for ax in axes:
                ax.axvline(onset, color=c, alpha=0.7, linewidth=1.2)
        axes[0].text(onsets[0], axes[0].get_ylim()[1] * 0.85,
                     cond, color=c, fontsize=8, rotation=90)

    plt.suptitle(f'{participant_id} — Sinal completo pós-pipeline SPM', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Sinal completo salvo em: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Pipeline SPM fNIRS replicada em Python para arquivos .nirs'
    )
    parser.add_argument('nirs_file', help='Caminho para o arquivo .nirs')
    parser.add_argument('--baseline_samples', type=int, default=100,
                        help='Número de amostras iniciais para baseline (padrão: 100)')
    parser.add_argument('--wavelet', type=str, default='sym8',
                        help='Família wavelet (padrão: sym8)')
    parser.add_argument('--wavelet_level', type=int, default=4,
                        help='Nível de decomposição wavelet (padrão: 4)')
    parser.add_argument('--tmin', type=float, default=-20,
                        help='Início da época em segundos (padrão: -20)')
    parser.add_argument('--tmax_epoch', type=float, default=60,
                        help='Fim da época em segundos (padrão: 60)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Diretório de saída para os gráficos')
    args = parser.parse_args()

    path = args.nirs_file
    if not os.path.exists(path):
        print(f"Arquivo não encontrado: {path}")
        sys.exit(1)

    participant_id = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PIPELINE SPM — {participant_id}")
    print(f"{'='*60}")

    # ── Etapa 1: Carregar ────────────────────────────────────────
    print("\n[1/6] Carregando .nirs...")
    oxy, dxy, events, n_ch = load_nirs(path)
    n_samples = oxy.shape[0]
    print(f"  Eventos detectados: {list(events.keys())}")

    # Usar durações padrão para condições conhecidas
    durations = {k: DEFAULT_DURATIONS.get(k, 60.0) for k in events.keys()}

    # ── Etapa 2: Baseline initial time ──────────────────────────
    print(f"\n[2/6] Baseline 'initial time' ({args.baseline_samples} amostras)...")
    oxy_bl, dxy_bl = baseline_initial_time(oxy, dxy, n_samples=args.baseline_samples)
    print(f"  HbO offset removido: {np.mean(oxy[:args.baseline_samples,:], axis=0).round(3)} µM")

    # ── Etapa 3: Wavelet-MDL ─────────────────────────────────────
    print(f"\n[3/6] Wavelet-MDL ({args.wavelet}, level={args.wavelet_level})...")
    oxy_wav = wavelet_mdl(oxy_bl, wavelet=args.wavelet, level=args.wavelet_level)
    dxy_wav = wavelet_mdl(dxy_bl, wavelet=args.wavelet, level=args.wavelet_level)
    std_before = np.std(oxy_bl, axis=0)
    std_after = np.std(oxy_wav, axis=0)
    reducao = (1 - std_after / std_before) * 100
    print(f"  Redução de std HbO por canal: {reducao.round(1)}%")

    # ── Etapa 4: Matriz de design GLM ───────────────────────────
    print(f"\n[4/6] Montando matriz de design GLM ({len(events)} condições + constante)...")
    X = build_design_matrix(n_samples, events, durations, fs=FS, hrf_dt=HRF_DT)
    print(f"  Matriz X: {X.shape} ({X.shape[0]} amostras × {X.shape[1]} regressores)")

    # ── Etapa 5: GLM OLS → resíduos ─────────────────────────────
    print(f"\n[5/6] Ajustando GLM (OLS) e extraindo resíduos...")
    oxy_final = glm_residuals(oxy_wav, X)
    dxy_final = glm_residuals(dxy_wav, X)
    std_res_oxy = np.std(oxy_final, axis=0)
    reducao_glm = (1 - std_res_oxy / std_after) * 100
    print(f"  Redução adicional de std HbO pelo GLM: {reducao_glm.round(1)}%")
    print(f"  HbO final range: [{oxy_final.min():.2f}, {oxy_final.max():.2f}] µM")
    print(f"  HbR final range: [{dxy_final.min():.2f}, {dxy_final.max():.2f}] µM")

    # ── Etapa 6: Extração de épocas e plots ─────────────────────
    print(f"\n[6/6] Extraindo épocas e gerando gráficos...")
    epochs_oxy, adj_oxy = extract_epochs(oxy_final, events,
                                         tmin=args.tmin, tmax=args.tmax_epoch, fs=FS)
    epochs_dxy, adj_dxy = extract_epochs(dxy_final, events,
                                         tmin=args.tmin, tmax=args.tmax_epoch, fs=FS)

    all_adj = {**adj_oxy, **adj_dxy}
    for cond, trials in all_adj.items():
        for t in trials:
            msgs = []
            if 'tmin_used' in t:
                msgs.append(f"pré-evento: {t['tmin_requested']:.1f}s → {t['tmin_used']:.1f}s")
            if 'tmax_used' in t:
                msgs.append(f"pós-evento: {t['tmax_requested']:.1f}s → {t['tmax_used']:.1f}s")
            print(f"  ⚠ Ajuste de janela — {cond} (trial {t['trial']}): {', '.join(msgs)}")

    # Salvar plots
    hrf_path = os.path.join(args.output_dir, f'{participant_id}_hrf.png')
    sig_path = os.path.join(args.output_dir, f'{participant_id}_sinal.png')

    plot_hrfs(epochs_oxy, epochs_dxy, hrf_path, participant_id)
    plot_signal_complete(oxy_final, dxy_final, events, sig_path, participant_id)

    # Salvar dados processados como .npy para uso posterior
    np.save(os.path.join(args.output_dir, f'{participant_id}_oxy_final.npy'), oxy_final)
    np.save(os.path.join(args.output_dir, f'{participant_id}_dxy_final.npy'), dxy_final)

    print(f"\n{'='*60}")
    print(f"  Pipeline concluída para {participant_id}")
    print(f"  Arquivos gerados em: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
