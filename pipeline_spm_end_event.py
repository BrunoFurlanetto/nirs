"""
pipeline_spm_end_event.py
--------------------------
Igual ao pipeline_spm.py, mas com suporte a um evento de finalização.

Uso:
    python pipeline_spm_end_event.py arquivo.nirs --end_event S
    python pipeline_spm_end_event.py arquivo.nirs --end_event S --baseline_samples 100

Argumento --end_event:
    Letra (ex: 'S', 'F', 'E') que identifica o evento de finalização.
    O sinal é truncado na primeira ocorrência de qualquer condição cujo nome
    começa com essa letra (ex.: 'S1', 'S2'). Todas as condições ficam
    restritas aos dados anteriores a esse corte.
    Se a letra não for encontrada nos eventos, o arquivo é processado inteiro
    com aviso.

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
from scipy.signal import butter, filtfilt


# ── Configurações padrão ──────────────────────────────────────────────────────
FS = 10.0
DPF = 6.61
HRF_DURATION = 32
HRF_DT = 0.1

DEFAULT_DURATIONS = {
    'A1': 60, 'B1': 60, 'C1': 60, 'D1': 60, 'E1': 60,
    'F1': 60, 'H1': 60, 'I1': 60, 'M1': 60, 'P1': 60, 'S1': 60,
}

COLORS = {
    'A1': 'red',    'B1': 'blue',   'C1': 'green',  'D1': 'orange',
    'E1': 'purple', 'F1': 'brown',  'H1': 'pink',   'I1': 'cyan',
    'M1': 'olive',  'P1': 'magenta','S1': 'gray',
}

EXTINCTION = {
    752: {'HbO': 1292.4, 'HbR': 3669.9},
    840: {'HbO': 2549.4, 'HbR': 1799.5},
}


# ── Carregar .nirs ────────────────────────────────────────────────────────────

def load_nirs(path):
    data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in data.keys() if not k.startswith('__')]
    print(f"  Chaves encontradas: {keys}")

    result = {'mode': None, 'events': {}, 'wavelengths': None, 'measlist': None}

    if 'SD' in data:
        SD = data['SD']
        result['wavelengths'] = list(np.atleast_1d(np.array(SD.Lambda)).astype(float))
        result['measlist'] = np.atleast_2d(np.array(SD.MeasList, dtype=int))

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

    elif 'd' in data:
        d = np.atleast_2d(data['d'])
        if d.shape[0] < d.shape[1]:
            d = d.T
        n_samples = d.shape[0]
        n_cols = d.shape[1]

        is_intensity = (d.min() >= 0) and (d.max() < 10)

        if is_intensity and result['measlist'] is not None:
            result['mode'] = 'intensity'
            result['intensity'] = d
            n_wl = len(result['wavelengths'])
            result['n_ch'] = n_cols // n_wl
            print(f"  Formato: INTENSIDADE bruta — {result['n_ch']} canais × "
                  f"{n_wl} λ = {n_cols} colunas")
        else:
            n_ch = n_cols // 2
            result['mode'] = 'concentration'
            result['oxy'] = d[:, :n_ch]
            result['dxy'] = d[:, n_ch:]
            result['n_ch'] = n_ch
            print(f"  Formato: concentração ('d' dividido) — {n_ch} canais")
    else:
        raise ValueError(f"Formato não reconhecido. Chaves: {keys}")

    print(f"  {n_samples} amostras @ {FS} Hz = {n_samples/FS:.1f}s")

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
    n_samples = intensity.shape[0]
    n_wl = len(wavelengths)
    n_ch = intensity.shape[1] // n_wl

    mean_int = np.mean(intensity, axis=0, keepdims=True)
    mean_int[mean_int <= 0] = 1e-12
    od = -np.log(intensity / mean_int)

    oxy = np.zeros((n_samples, n_ch))
    dxy = np.zeros((n_samples, n_ch))

    pairs = {}
    for col, row in enumerate(measlist):
        src, det, _, wl_idx = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        key = (src, det)
        pairs.setdefault(key, {})[wl_idx] = col

    for ch_idx, (key, wl_cols) in enumerate(pairs.items()):
        if ch_idx >= n_ch:
            break
        d_mm = distances_mm[ch_idx] if ch_idx < len(distances_mm) else 35.0
        d_cm = d_mm / 10.0

        ext_matrix = np.zeros((n_wl, 2))
        od_ch = np.zeros((n_samples, n_wl))
        L = np.zeros(n_wl)

        for wi, wl in enumerate(wavelengths):
            wl_int = int(round(wl))
            wl_key = min(EXTINCTION.keys(), key=lambda k: abs(k - wl_int))
            ext_matrix[wi, 0] = EXTINCTION[wl_key]['HbO']
            ext_matrix[wi, 1] = EXTINCTION[wl_key]['HbR']
            L[wi] = d_cm * dpf_values[wl]
            col = wl_cols.get(wi + 1)
            if col is not None:
                od_ch[:, wi] = od[:, col]

        A = ext_matrix * L[:, None]
        try:
            A_inv = np.linalg.pinv(A)
        except np.linalg.LinAlgError:
            continue
        conc = od_ch @ A_inv.T
        oxy[:, ch_idx] = conc[:, 0] * 1e6
        dxy[:, ch_idx] = conc[:, 1] * 1e6

    return oxy, dxy


def compute_channel_distances(path):
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

    unit = getattr(SD, 'SpatialUnit', 'mm')
    if not isinstance(unit, str):
        unit = 'mm'
    unit = unit.strip().lower()
    to_mm = {'mm': 1.0, 'cm': 10.0, 'm': 1000.0}.get(unit, 1.0)

    results = []
    seen = set()
    for row in meas:
        si, di = int(row[0]) - 1, int(row[1]) - 1
        if (si, di) in seen:
            continue
        seen.add((si, di))
        if si < len(src) and di < len(det):
            dist_native = float(np.linalg.norm(src[si] - det[di]))
            dist_mm = dist_native * to_mm
            results.append((f"S{si+1}-D{di+1}", dist_mm, unit))

    return results


# ── Evento de finalização ─────────────────────────────────────────────────────

def apply_end_event(oxy, dxy, events, end_letter, fs=FS):
    """
    Trunca sinal e eventos na primeira ocorrência global do evento-letra.

    Uso no CLI (script de linha de comando). Para o app Streamlit, use
    compute_durations_from_end_events() que preserva o sinal inteiro e calcula
    janelas individuais por trial.

    Retorna (oxy_cut, dxy_cut, events_cut, cut_time).
    """
    letter = end_letter.upper()
    matching = {
        name: onsets
        for name, onsets in events.items()
        if name.upper().startswith(letter)
    }

    if not matching:
        print(f"  [AVISO] Evento de finalização '{letter}' não encontrado nos dados. "
              f"Processando sinal inteiro.")
        return oxy, dxy, events, None

    cut_time = min(float(np.min(onsets)) for onsets in matching.values())
    cut_sample = int(cut_time * fs)

    print(f"  Evento de finalização '{letter}': corte em t={cut_time:.2f}s "
          f"(sample {cut_sample})")

    oxy_cut = oxy[:cut_sample, :]
    dxy_cut = dxy[:cut_sample, :]

    events_cut = {}
    for name, onsets in events.items():
        if name.upper().startswith(letter):
            continue
        valid = onsets[onsets < cut_time]
        if len(valid) > 0:
            events_cut[name] = valid

    remaining = list(events_cut.keys())
    removed_late = [
        name for name, onsets in events.items()
        if not name.upper().startswith(letter) and not np.any(onsets < cut_time)
    ]
    if removed_late:
        print(f"  Condições sem onset antes do corte (removidas): {removed_late}")
    print(f"  Condições mantidas: {remaining}")

    return oxy_cut, dxy_cut, events_cut, cut_time


def compute_durations_from_end_events(events, end_letter):
    """
    Para cada condição, acha o próximo onset do evento-letra APÓS cada trial
    e usa a diferença (next_F - onset) como duração daquele trial.

    Retorna:
      durations_per_cond  — dict {cond: mediana_das_duracoes_em_s}
      events_filtered     — events sem as condições de finalização
      f_onsets            — lista ordenada de todos os onsets da letra

    Se a letra não existir, retorna (None, events, []).
    """
    letter = end_letter.upper()

    f_onsets = []
    for name, onsets in events.items():
        if name.upper().startswith(letter):
            f_onsets.extend(np.atleast_1d(onsets).tolist())
    f_onsets = sorted(f_onsets)

    if not f_onsets:
        return None, events, []

    durations_per_cond = {}
    events_filtered = {}

    for name, onsets in events.items():
        if name.upper().startswith(letter):
            continue  # pula as condições de finalização
        events_filtered[name] = onsets

        trial_durs = []
        for onset in np.atleast_1d(onsets):
            next_f = next((f for f in f_onsets if f > onset), None)
            if next_f is not None:
                trial_durs.append(float(next_f - onset))

        if trial_durs:
            durations_per_cond[name] = float(np.median(trial_durs))

    print(f"  Duração por condição (onset → próximo '{letter}'): "
          + ", ".join(f"{k}={v:.1f}s" for k, v in durations_per_cond.items()))

    return durations_per_cond, events_filtered, f_onsets


# ── Baseline ──────────────────────────────────────────────────────────────────

def baseline_initial_time(oxy, dxy, n_samples=100):
    bl_oxy = np.mean(oxy[:n_samples, :], axis=0, keepdims=True)
    bl_dxy = np.mean(dxy[:n_samples, :], axis=0, keepdims=True)
    return oxy - bl_oxy, dxy - bl_dxy


# ── Wavelet-MDL ───────────────────────────────────────────────────────────────

def _mdl_threshold(coeffs):
    n = len(coeffs)
    abs_c = np.abs(coeffs)
    sorted_c = np.sort(abs_c)[::-1]

    mdl = np.inf
    best_k = 0

    for k in range(1, n):
        signal_power = np.sum(sorted_c[:k] ** 2)
        noise_var = np.sum(sorted_c[k:] ** 2) / max(n - k, 1)
        if noise_var <= 0:
            continue
        cost = (n - k) * np.log(noise_var) + k * np.log(signal_power / k + 1e-10) + k * np.log(n)
        if cost < mdl:
            mdl = cost
            best_k = k

    if best_k == 0 or best_k >= n:
        return sorted_c[0]
    return sorted_c[best_k]


def wavelet_mdl(signal, wavelet='sym8', level=4):
    n_samples, n_ch = signal.shape
    corrected = np.zeros_like(signal)

    for ch in range(n_ch):
        x = signal[:, ch].copy()
        n_pad = 2 ** int(np.ceil(np.log2(len(x))))
        x_pad = np.pad(x, (0, n_pad - len(x)), mode='reflect')
        coeffs = pywt.wavedec(x_pad, wavelet, level=level)
        coeffs_thresh = [coeffs[0]]
        for detail in coeffs[1:]:
            thr = _mdl_threshold(detail)
            detail_thresh = pywt.threshold(detail, thr, mode='soft')
            coeffs_thresh.append(detail_thresh)
        x_rec = pywt.waverec(coeffs_thresh, wavelet)[:len(x)]
        corrected[:, ch] = x_rec

    return corrected


# ── GLM com HRF canônica ──────────────────────────────────────────────────────

def spm_hrf(dt=0.1, duration=32):
    t = np.arange(0, duration, dt)
    p1, p2 = 6.0, 16.0
    d1, d2 = 1.0, 1.0
    ratio = 6.0

    def gamma_pdf(t, shape, scale):
        return (t ** (shape - 1) * np.exp(-t / scale) /
                (scale ** shape * gamma(shape)))

    h = (gamma_pdf(t, p1 / d1, d1) - gamma_pdf(t, p2 / d2, d2) / ratio)
    h = h / np.max(np.abs(h))
    return h


def build_design_matrix(n_samples, events, durations, fs=10.0, hrf_dt=0.1):
    hrf = spm_hrf(dt=hrf_dt, duration=HRF_DURATION)
    n_conds = len(events)
    X = np.zeros((n_samples, n_conds + 1))

    for col, (cond_name, onsets) in enumerate(events.items()):
        duration = durations.get(cond_name, 60.0)
        n_hires = int(n_samples / fs / hrf_dt)
        stim = np.zeros(n_hires)

        for onset in onsets:
            i_start = int(onset / hrf_dt)
            i_end = int((onset + duration) / hrf_dt)
            i_start = max(0, min(i_start, n_hires))
            i_end = max(0, min(i_end, n_hires))
            stim[i_start:i_end] = 1.0

        convolved = np.convolve(stim, hrf)[:n_hires]
        step = int(1.0 / (fs * hrf_dt))
        downsampled = convolved[::step]
        n = min(len(downsampled), n_samples)
        X[:n, col] = downsampled[:n]

    X[:, -1] = 1.0
    return X


def glm_residuals(signal, X):
    n_samples, n_ch = signal.shape
    residuals = np.zeros_like(signal)

    for ch in range(n_ch):
        y = signal[:, ch]
        beta, _, _, _ = lstsq(X, y)
        y_hat = X @ beta
        residuals[:, ch] = y - y_hat

    return residuals


# ── Extração de épocas ────────────────────────────────────────────────────────

def extract_epochs(signal, events, tmin=-20, tmax=60, fs=10.0, baseline=(-2, 0),
                   durations=None, post_margin=15.0,
                   tmin_per_cond=None, baseline_per_cond=None):
    epochs_data = {}
    adjustments = {}
    n_sig = len(signal)

    for cond_name, onsets in events.items():
        cond_tmin = tmin_per_cond.get(cond_name, tmin) if tmin_per_cond else tmin
        cond_baseline = baseline_per_cond.get(cond_name, baseline) if baseline_per_cond else baseline

        if durations is not None and cond_name in durations:
            cond_tmax = float(durations[cond_name]) + post_margin
        else:
            cond_tmax = tmax

        n_pre = int(round(abs(cond_tmin) * fs))
        n_post = int(round(cond_tmax * fs))
        n_samples_epoch = n_pre + n_post
        times = cond_tmin + np.arange(n_samples_epoch) / fs

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
                adj['tmin_requested'] = cond_tmin
                adj['tmin_used'] = round(-(i_onset / fs), 3)
            if pad_after > 0:
                epoch = np.pad(epoch, ((0, pad_after), (0, 0)), mode='constant')
                adj['tmax_requested'] = cond_tmax
                adj['tmax_used'] = round((n_sig - i_onset) / fs, 3)

            if adj:
                adj['trial'] = trial_idx + 1
                cond_adj.append(adj)

            bl_s = int((cond_baseline[0] - cond_tmin) * fs)
            bl_e = int((cond_baseline[1] - cond_tmin) * fs)
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
                'epochs': epochs,
            }

    return epochs_data, adjustments


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_hrfs(epochs_oxy, epochs_dxy, output_path, participant_id):
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

    plt.suptitle(f'{participant_id} — HRF (Pipeline SPM + corte por evento)\n'
                 f'Wavelet-MDL → Baseline initial time → GLM resíduos', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  HRFs salvas em: {output_path}")


def plot_signal_complete(oxy_final, dxy_final, events, output_path, participant_id,
                         cut_time=None, end_letter=None):
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

    ev_colors = list(COLORS.values())
    for i, (cond, onsets) in enumerate(events.items()):
        c = COLORS.get(cond, ev_colors[i % len(ev_colors)])
        for onset in onsets:
            for ax in axes:
                ax.axvline(onset, color=c, alpha=0.7, linewidth=1.2)
        axes[0].text(onsets[0], axes[0].get_ylim()[1] * 0.85,
                     cond, color=c, fontsize=8, rotation=90)

    if cut_time is not None:
        label = f'Fim ({end_letter})' if end_letter else 'Corte'
        for ax in axes:
            ax.axvline(cut_time, color='black', linestyle=':', linewidth=2.0,
                       label=label)
        axes[0].legend(fontsize=7, loc='upper right', ncol=4)

    title_suffix = f' | cortado em t={cut_time:.1f}s ({end_letter})' if cut_time else ''
    plt.suptitle(f'{participant_id} — Sinal completo pós-pipeline SPM{title_suffix}',
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Sinal completo salvo em: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Pipeline SPM fNIRS com evento de finalização configurável'
    )
    parser.add_argument('nirs_file', help='Caminho para o arquivo .nirs')
    parser.add_argument('--end_event', type=str, default=None,
                        help='Letra do evento de finalização (ex: S, F, E). '
                             'Trunca o sinal na primeira ocorrência desse evento.')
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
    print(f"  PIPELINE SPM (end_event) — {participant_id}")
    if args.end_event:
        print(f"  Evento de finalização: '{args.end_event.upper()}'")
    print(f"{'='*60}")

    # ── Etapa 1: Carregar ────────────────────────────────────────
    print("\n[1/7] Carregando .nirs...")
    nirs = load_nirs(path)

    if nirs['mode'] == 'intensity':
        print("  Modo intensidade detectado — convertendo via Beer-Lambert...")
        distances_info = compute_channel_distances(path)
        distances_mm = [d for _, d, _ in distances_info] if distances_info else []
        dpf_values = {wl: DPF for wl in nirs['wavelengths']}
        oxy, dxy = intensity_to_concentration(
            nirs['intensity'], nirs['measlist'],
            nirs['wavelengths'], distances_mm, dpf_values
        )
    else:
        oxy = nirs['oxy']
        dxy = nirs['dxy']

    events = nirs['events']

    # ── Etapa 2: Corte por evento de finalização ─────────────────
    cut_time = None
    if args.end_event:
        print(f"\n[2/7] Aplicando corte por evento '{args.end_event.upper()}'...")
        oxy, dxy, events, cut_time = apply_end_event(
            oxy, dxy, events, args.end_event, fs=FS
        )
    else:
        print("\n[2/7] Sem evento de finalização — sinal inteiro será processado.")

    n_samples = oxy.shape[0]
    print(f"  Amostras após corte: {n_samples} ({n_samples/FS:.1f}s)")

    if not events:
        print("  [ERRO] Nenhuma condição restante após o corte. Encerrando.")
        sys.exit(1)

    durations = {k: DEFAULT_DURATIONS.get(k, 60.0) for k in events.keys()}

    # ── Etapa 3: Baseline initial time ──────────────────────────
    print(f"\n[3/7] Baseline 'initial time' ({args.baseline_samples} amostras)...")
    oxy_bl, dxy_bl = baseline_initial_time(oxy, dxy, n_samples=args.baseline_samples)

    # ── Etapa 4: Wavelet-MDL ─────────────────────────────────────
    print(f"\n[4/7] Wavelet-MDL ({args.wavelet}, level={args.wavelet_level})...")
    oxy_wav = wavelet_mdl(oxy_bl, wavelet=args.wavelet, level=args.wavelet_level)
    dxy_wav = wavelet_mdl(dxy_bl, wavelet=args.wavelet, level=args.wavelet_level)
    std_before = np.std(oxy_bl, axis=0)
    std_after = np.std(oxy_wav, axis=0)
    reducao = (1 - std_after / std_before) * 100
    print(f"  Redução de std HbO por canal: {reducao.round(1)}%")

    # ── Etapa 5: Matriz de design GLM ───────────────────────────
    print(f"\n[5/7] Montando matriz de design GLM ({len(events)} condições + constante)...")
    X = build_design_matrix(n_samples, events, durations, fs=FS, hrf_dt=HRF_DT)
    print(f"  Matriz X: {X.shape}")

    # ── Etapa 6: GLM OLS → resíduos ─────────────────────────────
    print(f"\n[6/7] Ajustando GLM (OLS) e extraindo resíduos...")
    oxy_final = glm_residuals(oxy_wav, X)
    dxy_final = glm_residuals(dxy_wav, X)
    print(f"  HbO final range: [{oxy_final.min():.2f}, {oxy_final.max():.2f}] µM")
    print(f"  HbR final range: [{dxy_final.min():.2f}, {dxy_final.max():.2f}] µM")

    # ── Etapa 7: Épocas e plots ──────────────────────────────────
    print(f"\n[7/7] Extraindo épocas e gerando gráficos...")
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

    end_suffix = f'_ate_{args.end_event.upper()}' if args.end_event else ''
    hrf_path = os.path.join(args.output_dir, f'{participant_id}{end_suffix}_hrf.png')
    sig_path = os.path.join(args.output_dir, f'{participant_id}{end_suffix}_sinal.png')

    plot_hrfs(epochs_oxy, epochs_dxy, hrf_path, participant_id)
    plot_signal_complete(oxy_final, dxy_final, events, sig_path, participant_id,
                         cut_time=cut_time, end_letter=args.end_event)

    np.save(os.path.join(args.output_dir, f'{participant_id}{end_suffix}_oxy_final.npy'), oxy_final)
    np.save(os.path.join(args.output_dir, f'{participant_id}{end_suffix}_dxy_final.npy'), dxy_final)

    print(f"\n{'='*60}")
    print(f"  Pipeline concluída para {participant_id}")
    if cut_time:
        print(f"  Sinal processado: 0 → {cut_time:.1f}s (evento '{args.end_event.upper()}')")
    print(f"  Arquivos gerados em: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
