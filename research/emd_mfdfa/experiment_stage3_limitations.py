"""
ЭТАП 3: Исследование ограничений
=================================

Цель: Изучить влияние длины ряда и уровня шума на точность метода.

Эксперименты:
1. Влияние длины ряда N (при H=0.7, без шума)
2. Влияние уровня шума SNR (при H=0.7, N=2048)
"""

import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from StatTools.analysis.emd_mfdfa import estimate_hurst_emd
from StatTools.generators.kasdin_generator import create_kasdin_generator

print("=" * 80)
print(" " * 15 + "Этап 3: Исследование ограничений")
print("=" * 80)

# ЭКСПЕРИМЕНТ 1: Влияние длины ряда

print("\n" + "─" * 80)
print("Эксперимент 1: Влияние длины ряда N")
print("─" * 80)

H_TRUE = 0.7
N_VALUES = [512, 1024, 2048, 4096, 8192]
N_REALIZATIONS = 10
DEGREE = 2

print(f"\nПараметры:")
print(f"  H (фикс.): {H_TRUE}")
print(f"  N значения: {N_VALUES}")
print(f"  Реализаций: {N_REALIZATIONS}")

results_exp1 = {
    "parameters": {
        "H_true": H_TRUE,
        "N_values": N_VALUES,
        "n_realizations": N_REALIZATIONS,
        "degree": DEGREE,
    },
    "experiments": [],
}

for N in N_VALUES:
    print(f"\n  Тестирование N = {N}...")

    h_estimates = []

    for i in tqdm(range(N_REALIZATIONS), desc=f"N={N}", ncols=70):
        try:
            seed = 7000 + N + i
            generator = create_kasdin_generator(h=H_TRUE, length=N, normalize=True)
            np.random.seed(seed)
            signal = generator.get_full_sequence()

            H_est = estimate_hurst_emd(signal, degree=DEGREE)
            h_estimates.append(H_est)

            results_exp1["experiments"].append(
                {
                    "N": N,
                    "realization": i,
                    "H_estimated": float(H_est),
                    "error": float(abs(H_est - H_TRUE)),
                }
            )

        except Exception as e:
            print(f"    Ошибка: {e}")
            h_estimates.append(np.nan)

    # Статистика
    h_estimates = np.array(h_estimates)
    valid = ~np.isnan(h_estimates)

    if np.sum(valid) > 0:
        mean_est = np.mean(h_estimates[valid])
        std_est = np.std(h_estimates[valid])
        mean_error = np.mean(np.abs(h_estimates[valid] - H_TRUE))

        print(f"    H: {mean_est:.4f} ± {std_est:.4f}, MAE: {mean_error:.4f}")

        results_exp1[f"summary_N_{N}"] = {
            "N": N,
            "mean_estimate": float(mean_est),
            "std_estimate": float(std_est),
            "mean_error": float(mean_error),
        }

# Эксперимент 2: Влияние шума (SNR)

print("\n" + "─" * 80)
print("Эксперимент 2: Влияние уровня шума (SNR)")
print("─" * 80)

N = 2048
SNR_VALUES = [np.inf, 30, 20, 10, 5]  # dB (inf = без шума)

print(f"\nПараметры:")
print(f"  H (фикс.): {H_TRUE}")
print(f"  N (фикс.): {N}")
print(f"  SNR значения (dB): {SNR_VALUES}")
print(f"  Реализаций: {N_REALIZATIONS}")

results_exp2 = {
    "parameters": {
        "H_true": H_TRUE,
        "N": N,
        "SNR_values": SNR_VALUES,
        "n_realizations": N_REALIZATIONS,
        "degree": DEGREE,
    },
    "experiments": [],
}

for SNR in SNR_VALUES:
    snr_label = "∞" if SNR == np.inf else f"{SNR}dB"
    print(f"\n  Тестирование SNR = {snr_label}...")

    h_estimates = []

    for i in tqdm(range(N_REALIZATIONS), desc=f"SNR={snr_label}", ncols=70):
        try:
            seed = 8000 + int(SNR if SNR != np.inf else 999) + i
            generator = create_kasdin_generator(h=H_TRUE, length=N, normalize=True)
            np.random.seed(seed)
            signal = generator.get_full_sequence()

            # Добавление шума
            if SNR != np.inf:
                signal_power = np.mean(signal**2)
                noise_power = signal_power / (10 ** (SNR / 10))
                noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
                signal = signal + noise

            H_est = estimate_hurst_emd(signal, degree=DEGREE)
            h_estimates.append(H_est)

            results_exp2["experiments"].append(
                {
                    "SNR": float(SNR) if SNR != np.inf else None,
                    "realization": i,
                    "H_estimated": float(H_est),
                    "error": float(abs(H_est - H_TRUE)),
                }
            )

        except Exception as e:
            print(f"    Ошибка: {e}")
            h_estimates.append(np.nan)

    # Статистика
    h_estimates = np.array(h_estimates)
    valid = ~np.isnan(h_estimates)

    if np.sum(valid) > 0:
        mean_est = np.mean(h_estimates[valid])
        std_est = np.std(h_estimates[valid])
        mean_error = np.mean(np.abs(h_estimates[valid] - H_TRUE))

        print(f"    H: {mean_est:.4f} ± {std_est:.4f}, MAE: {mean_error:.4f}")

        snr_key = "inf" if SNR == np.inf else str(int(SNR))
        results_exp2[f"summary_SNR_{snr_key}"] = {
            "SNR": float(SNR) if SNR != np.inf else None,
            "mean_estimate": float(mean_est),
            "std_estimate": float(std_est),
            "mean_error": float(mean_error),
        }

# сохранение результатов

output_dir = Path(__file__).parent / "results"
output_dir.mkdir(exist_ok=True)

# Эксперимент 1
json_file1 = output_dir / "stage3_exp1_length_effect.json"
with open(json_file1, "w", encoding="utf-8") as f:
    json.dump(results_exp1, f, indent=2, ensure_ascii=False)

# Эксперимент 2
json_file2 = output_dir / "stage3_exp2_noise_effect.json"
with open(json_file2, "w", encoding="utf-8") as f:
    json.dump(results_exp2, f, indent=2, ensure_ascii=False)

# Текстовый отчёт
txt_file = output_dir / "stage3_limitations_results.txt"
with open(txt_file, "w", encoding="utf-8") as f:
    f.write("Этап 3: Исследование ограничений\n")
    f.write("=" * 80 + "\n\n")

    f.write("Эксперимент 1: Влияние длины ряда\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'N':>10} {'H_mean':>10} {'H_std':>10} {'MAE':>10}\n")
    f.write("-" * 80 + "\n")

    for N in N_VALUES:
        key = f"summary_N_{N}"
        if key in results_exp1:
            s = results_exp1[key]
            f.write(
                f"{s['N']:>10} {s['mean_estimate']:>10.4f} "
                f"{s['std_estimate']:>10.4f} {s['mean_error']:>10.4f}\n"
            )

    f.write("\n\nЭксперимент 2: Влияние уровня шума\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'SNR (dB)':>10} {'H_mean':>10} {'H_std':>10} {'MAE':>10}\n")
    f.write("-" * 80 + "\n")

    for SNR in SNR_VALUES:
        snr_key = "inf" if SNR == np.inf else str(int(SNR))
        key = f"summary_SNR_{snr_key}"
        if key in results_exp2:
            s = results_exp2[key]
            snr_label = "∞" if s["SNR"] is None else f"{int(s['SNR'])}"
            f.write(
                f"{snr_label:>10} {s['mean_estimate']:>10.4f} "
                f"{s['std_estimate']:>10.4f} {s['mean_error']:>10.4f}\n"
            )

print("\n" + "=" * 80)
print("Этап 3 завершён")
print("=" * 80)
print(f"\nРезультаты сохранены:")
print(f"  Эксп. 1: {json_file1}")
print(f"  Эксп. 2: {json_file2}")
print(f"  Отчёт:   {txt_file}")
