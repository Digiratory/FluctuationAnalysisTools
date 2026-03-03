"""
ЭТАП 2: Основной эксперимент - Систематические тесты

Цель: Оценить точность метода EMD-MFDFA на синтетических данных
с различными параметрами.

План:
- H ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
- N = 2048 (фиксировано для базового эксперимента)
- Количество реализаций: 10 для каждого H
- Метрика: MAE(H_estimated, H_true)
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
print(" " * 15 + "Этап 2: Систематические эксперименты")
print("=" * 80)

# Параметры эксперимента
H_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N = 2048
N_REALIZATIONS = 10
DEGREE = 2

print(f"\nПараметры:")
print(f"  H значения: {H_VALUES}")
print(f"  Длина сигнала: {N}")
print(f"  Реализаций для каждого H: {N_REALIZATIONS}")
print(f"  Степень детрендинга: {DEGREE}")
print(f"  Всего экспериментов: {len(H_VALUES) * N_REALIZATIONS}")

# Результаты
results = {
    "parameters": {
        "H_values": H_VALUES,
        "N": N,
        "n_realizations": N_REALIZATIONS,
        "degree": DEGREE,
    },
    "experiments": [],
}

# Счётчик
total_experiments = len(H_VALUES) * N_REALIZATIONS
current = 0

print("\n" + "=" * 80)
print("Запуск экспериментов")
print("=" * 80)

# Для каждого значения H
for H_true in H_VALUES:
    print(f"\n{'─' * 80}")
    print(
        f"H = {H_true} ({'антиперс.' if H_true < 0.5 else 'случ. блужд.' if H_true == 0.5 else 'персистент.'})"
    )
    print(f"{'─' * 80}")

    h_estimates = []

    # Прогресс-бар для реализаций
    pbar = tqdm(range(N_REALIZATIONS), desc=f"H={H_true}", ncols=80)

    for i in pbar:
        current += 1

        try:
            # Генерация сигнала
            seed = 1000 * int(H_true * 10) + i  # Уникальный seed
            generator = create_kasdin_generator(h=H_true, length=N, normalize=True)
            np.random.seed(seed)  # Для воспроизводимости
            signal = generator.get_full_sequence()

            # Анализ
            H_est = estimate_hurst_emd(signal, degree=DEGREE)
            h_estimates.append(H_est)

            # Сохранение результата
            results["experiments"].append(
                {
                    "H_true": H_true,
                    "realization": i,
                    "seed": seed,
                    "H_estimated": float(H_est),
                    "error": float(abs(H_est - H_true)),
                }
            )

            # Обновление прогресс-бара
            pbar.set_postfix(
                {"H_est": f"{H_est:.3f}", "err": f"{abs(H_est - H_true):.3f}"}
            )

        except Exception as e:
            print(f"\n  Ошибка в реализации {i}: {e}")
            h_estimates.append(np.nan)

    # Статистика для текущего H
    h_estimates = np.array(h_estimates)
    valid = ~np.isnan(h_estimates)

    if np.sum(valid) > 0:
        mean_est = np.mean(h_estimates[valid])
        std_est = np.std(h_estimates[valid])
        mean_error = np.mean(np.abs(h_estimates[valid] - H_true))
        std_error = np.std(np.abs(h_estimates[valid] - H_true))

        print(f"\n  Результаты для H = {H_true}:")
        print(f"    Средняя оценка H: {mean_est:.4f} ± {std_est:.4f}")
        print(f"    Истинное H:       {H_true:.4f}")
        print(f"    MAE:              {mean_error:.4f} ± {std_error:.4f}")
        print(f"    Успешных:         {np.sum(valid)}/{N_REALIZATIONS}")

        # Сохранение сводки
        results[f"summary_H_{H_true}"] = {
            "H_true": H_true,
            "mean_estimate": float(mean_est),
            "std_estimate": float(std_est),
            "mean_error": float(mean_error),
            "std_error": float(std_error),
            "n_successful": int(np.sum(valid)),
        }

# Общая статистика
print("\n" + "=" * 80)
print(" " * 25 + "ИТОГОВАЯ СТАТИСТИКА")
print("=" * 80)

print(
    f"\n{'H истинное':>12} {'H среднее':>12} {'Станд. откл.':>14} {'MAE':>10} {'σ(MAE)':>10}"
)
print("─" * 80)

all_errors = []
for H_true in H_VALUES:
    key = f"summary_H_{H_true}"
    if key in results:
        s = results[key]
        print(
            f"{s['H_true']:>12.2f} {s['mean_estimate']:>12.4f} "
            f"{s['std_estimate']:>14.4f} {s['mean_error']:>10.4f} "
            f"{s['std_error']:>10.4f}"
        )
        all_errors.append(s["mean_error"])

print("─" * 80)
if all_errors:
    print(f"{'Общая MAE:':>40} {np.mean(all_errors):>10.4f}")
    print(f"{'Макс. MAE:':>40} {np.max(all_errors):>10.4f}")
    print(f"{'Мин. MAE:':>40} {np.min(all_errors):>10.4f}")

# Сохранение результатов
output_dir = Path(__file__).parent / "results"
output_dir.mkdir(exist_ok=True)

# JSON с полными результатами
json_file = output_dir / "stage2_systematic_results.json"
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Текстовый отчёт
txt_file = output_dir / "stage2_systematic_results.txt"
with open(txt_file, "w", encoding="utf-8") as f:
    f.write("Этап 2: Систематические эксперименты\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Параметры:\n")
    f.write(f"  N = {N}\n")
    f.write(f"  Реализаций: {N_REALIZATIONS}\n")
    f.write(f"  Степень: {DEGREE}\n\n")
    f.write(
        f"{'H_true':>10} {'H_mean':>10} {'H_std':>10} {'MAE':>10} {'MAE_std':>10}\n"
    )
    f.write("-" * 80 + "\n")

    for H_true in H_VALUES:
        key = f"summary_H_{H_true}"
        if key in results:
            s = results[key]
            f.write(
                f"{s['H_true']:>10.2f} {s['mean_estimate']:>10.4f} "
                f"{s['std_estimate']:>10.4f} {s['mean_error']:>10.4f} "
                f"{s['std_error']:>10.4f}\n"
            )

    f.write("\n")
    if all_errors:
        f.write(f"Общая MAE: {np.mean(all_errors):.4f}\n")
        f.write(f"Макс. MAE: {np.max(all_errors):.4f}\n")
        f.write(f"Мин. MAE: {np.min(all_errors):.4f}\n")

print("\n" + "=" * 80)
print("Этап 2 завершён")
print("=" * 80)
print(f"\nРезультаты сохранены:")
print(f"  JSON: {json_file}")
print(f"  TXT:  {txt_file}")
