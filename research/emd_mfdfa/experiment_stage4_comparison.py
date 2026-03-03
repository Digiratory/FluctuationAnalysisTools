"""
ЭТАП 4: Сравнительный анализ
=============================

Цель: Сравнить EMD-MFDFA с классическим DFA.

План:
- Тестирование обоих методов на одних и тех же сигналах
- H ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
- N = 2048
- Реализаций: 10 для каждого H
- Метрики: MAE, время выполнения, устойчивость
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from StatTools.analysis.dfa import DFA
from StatTools.analysis.emd_mfdfa import estimate_hurst_emd
from StatTools.generators.kasdin_generator import create_kasdin_generator

print("=" * 80)
print(" " * 15 + "Этап 4: Сравнительный анализ")
print(" " * 20 + "EMD-MFDFA vs DFA")
print("=" * 80)

# Параметры
H_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N = 2048
N_REALIZATIONS = 10
DEGREE = 2

print(f"\nПараметры:")
print(f"  H значения: {H_VALUES}")
print(f"  Длина сигнала: {N}")
print(f"  Реализаций: {N_REALIZATIONS}")
print(f"  Степень детрендинга: {DEGREE}")

results = {
    "parameters": {
        "H_values": H_VALUES,
        "N": N,
        "n_realizations": N_REALIZATIONS,
        "degree": DEGREE,
    },
    "experiments": [],
}

print("\n" + "=" * 80)
print("Запуск сравнительных экспериментов")
print("=" * 80)

for H_true in H_VALUES:
    print(f"\n{'─' * 80}")
    print(f"H = {H_true}")
    print(f"{'─' * 80}")

    emd_estimates = []
    dfa_estimates = []
    emd_times = []
    dfa_times = []

    pbar = tqdm(range(N_REALIZATIONS), desc=f"H={H_true}", ncols=80)

    for i in pbar:
        try:
            # Генерация сигнала (одинакового для обоих методов)
            seed = 9000 + int(H_true * 10) + i
            generator = create_kasdin_generator(h=H_true, length=N, normalize=True)
            np.random.seed(seed)
            signal = generator.get_full_sequence()

            # ===== EMD-MFDFA =====
            start_time = time.time()
            H_emd = estimate_hurst_emd(signal, degree=DEGREE)
            emd_time = time.time() - start_time
            emd_estimates.append(H_emd)
            emd_times.append(emd_time)

            # ===== Классический DFA =====
            start_time = time.time()
            dfa = DFA(signal, degree=DEGREE, root=False)
            H_dfa = dfa.find_h()
            dfa_time = time.time() - start_time
            dfa_estimates.append(H_dfa)
            dfa_times.append(dfa_time)

            # Сохранение результата
            results["experiments"].append(
                {
                    "H_true": H_true,
                    "realization": i,
                    "seed": seed,
                    "emd_estimate": float(H_emd),
                    "emd_error": float(abs(H_emd - H_true)),
                    "emd_time": float(emd_time),
                    "dfa_estimate": float(H_dfa),
                    "dfa_error": float(abs(H_dfa - H_true)),
                    "dfa_time": float(dfa_time),
                }
            )

            pbar.set_postfix({"EMD": f"{H_emd:.3f}", "DFA": f"{H_dfa:.3f}"})

        except Exception as e:
            print(f"\n  Ошибка в реализации {i}: {e}")
            emd_estimates.append(np.nan)
            dfa_estimates.append(np.nan)

    # Статистика для текущего H
    emd_estimates = np.array(emd_estimates)
    dfa_estimates = np.array(dfa_estimates)
    emd_times = np.array(emd_times)
    dfa_times = np.array(dfa_times)

    emd_valid = ~np.isnan(emd_estimates)
    dfa_valid = ~np.isnan(dfa_estimates)

    if np.sum(emd_valid) > 0 and np.sum(dfa_valid) > 0:
        # EMD-MFDFA статистика
        emd_mean = np.mean(emd_estimates[emd_valid])
        emd_std = np.std(emd_estimates[emd_valid])
        emd_mae = np.mean(np.abs(emd_estimates[emd_valid] - H_true))
        emd_time_mean = np.mean(emd_times[emd_valid])

        # DFA статистика
        dfa_mean = np.mean(dfa_estimates[dfa_valid])
        dfa_std = np.std(dfa_estimates[dfa_valid])
        dfa_mae = np.mean(np.abs(dfa_estimates[dfa_valid] - H_true))
        dfa_time_mean = np.mean(dfa_times[dfa_valid])

        print(f"\n  Результаты для H = {H_true}:")
        print(
            f"    EMD-MFDFA: H = {emd_mean:.4f} ± {emd_std:.4f}, MAE = {emd_mae:.4f}, время = {emd_time_mean:.2f}с"
        )
        print(
            f"    DFA:       H = {dfa_mean:.4f} ± {dfa_std:.4f}, MAE = {dfa_mae:.4f}, время = {dfa_time_mean:.2f}с"
        )

        # Определение победителя
        if emd_mae < dfa_mae:
            winner = "EMD-MFDFA"
            improvement = ((dfa_mae - emd_mae) / dfa_mae) * 100
        else:
            winner = "DFA"
            improvement = ((emd_mae - dfa_mae) / emd_mae) * 100

        print(f"    Лучший метод: {winner} (улучшение на {improvement:.1f}%)")

        results[f"summary_H_{H_true}"] = {
            "H_true": H_true,
            "emd": {
                "mean": float(emd_mean),
                "std": float(emd_std),
                "mae": float(emd_mae),
                "time_mean": float(emd_time_mean),
            },
            "dfa": {
                "mean": float(dfa_mean),
                "std": float(dfa_std),
                "mae": float(dfa_mae),
                "time_mean": float(dfa_time_mean),
            },
            "winner": winner,
            "improvement_percent": float(improvement),
        }

# Общая статистика
print("\n" + "=" * 80)
print(" " * 25 + "ИТОГОВОЕ СРАВНЕНИЕ")
print("=" * 80)

print(
    f"\n{'H':>6} {'EMD MAE':>10} {'DFA MAE':>10} {'EMD время':>12} {'DFA время':>12} {'Лучший':>12}"
)
print("─" * 80)

emd_total_mae = []
dfa_total_mae = []
emd_wins = 0
dfa_wins = 0

for H_true in H_VALUES:
    key = f"summary_H_{H_true}"
    if key in results:
        s = results[key]
        print(
            f"{s['H_true']:>6.2f} {s['emd']['mae']:>10.4f} {s['dfa']['mae']:>10.4f} "
            f"{s['emd']['time_mean']:>11.2f}s {s['dfa']['time_mean']:>11.2f}s "
            f"{s['winner']:>12}"
        )

        emd_total_mae.append(s["emd"]["mae"])
        dfa_total_mae.append(s["dfa"]["mae"])

        if s["winner"] == "EMD-MFDFA":
            emd_wins += 1
        else:
            dfa_wins += 1

print("─" * 80)
if emd_total_mae and dfa_total_mae:
    emd_avg = np.mean(emd_total_mae)
    dfa_avg = np.mean(dfa_total_mae)
    print(f"{'Средняя MAE:':>20} EMD={emd_avg:.4f}, DFA={dfa_avg:.4f}")
    print(f"{'Побед:':>20} EMD={emd_wins}, DFA={dfa_wins}")

    if emd_avg < dfa_avg:
        overall_improvement = ((dfa_avg - emd_avg) / dfa_avg) * 100
        print(f"\n{'Итог:':>20} EMD-MFDFA лучше на {overall_improvement:.1f}%")
    else:
        overall_improvement = ((emd_avg - dfa_avg) / emd_avg) * 100
        print(f"\n{'Итог:':>20} DFA лучше на {overall_improvement:.1f}%")

# Сохранение результатов
output_dir = Path(__file__).parent / "results"
output_dir.mkdir(exist_ok=True)

json_file = output_dir / "stage4_comparison_results.json"
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

txt_file = output_dir / "stage4_comparison_results.txt"
with open(txt_file, "w", encoding="utf-8") as f:
    f.write("Этап 4: Сравнительный анализ (EMD-MFDFA vs DFA)\n")
    f.write("=" * 80 + "\n\n")
    f.write(
        f"{'H_true':>8} {'EMD_MAE':>10} {'DFA_MAE':>10} {'EMD_time':>10} {'DFA_time':>10} {'Winner':>12}\n"
    )
    f.write("-" * 80 + "\n")

    for H_true in H_VALUES:
        key = f"summary_H_{H_true}"
        if key in results:
            s = results[key]
            f.write(
                f"{s['H_true']:>8.2f} {s['emd']['mae']:>10.4f} {s['dfa']['mae']:>10.4f} "
                f"{s['emd']['time_mean']:>10.2f} {s['dfa']['time_mean']:>10.2f} "
                f"{s['winner']:>12}\n"
            )

    f.write("\n")
    if emd_total_mae and dfa_total_mae:
        f.write(f"Средняя MAE:\n")
        f.write(f"  EMD-MFDFA: {np.mean(emd_total_mae):.4f}\n")
        f.write(f"  DFA:       {np.mean(dfa_total_mae):.4f}\n\n")
        f.write(f"Побед:\n")
        f.write(f"  EMD-MFDFA: {emd_wins}\n")
        f.write(f"  DFA:       {dfa_wins}\n")

print("\n" + "=" * 80)
print("Этап 4 завершён")
print("=" * 80)
print(f"\nРезультаты сохранены:")
print(f"  JSON: {json_file}")
print(f"  TXT:  {txt_file}")
