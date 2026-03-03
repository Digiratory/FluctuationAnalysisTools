"""
Мастер-скрипт для запуска всех экспериментов
=============================================

Запускает все этапы эксперимента последовательно:
- Этап 2: Систематические тесты
- Этап 3: Исследование ограничений
- Этап 4: Сравнительный анализ

После завершения создаёт итоговый отчёт.
"""

import subprocess
import sys
import time
from pathlib import Path

print("=" * 80)
print(" " * 20 + "Запуск всех экспериментов")
print(" " * 25 + "EMD-MFDFA")
print("=" * 80)

base_dir = Path(__file__).parent

experiments = [
    ("ЭТАП 2: Систематические эксперименты", "experiment_stage2_systematic.py"),
    ("ЭТАП 3: Исследование ограничений", "experiment_stage3_limitations.py"),
    ("ЭТАП 4: Сравнительный анализ", "experiment_stage4_comparison.py"),
]

results = {}
total_start_time = time.time()

for stage_name, script_name in experiments:
    print(f"\n{'=' * 80}")
    print(f"Запуск: {stage_name}")
    print(f"Скрипт: {script_name}")
    print(f"{'=' * 80}\n")

    script_path = base_dir / script_name
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            cwd=str(base_dir.parent.parent),
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"\n {stage_name} завершён успешно")
            print(f"  Время выполнения: {elapsed:.1f} секунд")
            results[stage_name] = {"status": "success", "time": elapsed}
        else:
            print(f"\n{stage_name} завершился с ошибкой")
            print(f"  Код возврата: {result.returncode}")
            results[stage_name] = {
                "status": "error",
                "time": elapsed,
                "returncode": result.returncode,
            }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\nОшибка при запуске {stage_name}: {e}")
        results[stage_name] = {"status": "exception", "time": elapsed, "error": str(e)}

total_elapsed = time.time() - total_start_time

print("\n" + "=" * 80)
print(" " * 25 + "Итоговая сводка")
print("=" * 80)

for stage_name, info in results.items():
    status_icon = "✓" if info["status"] == "success" else "✗"
    print(f"{status_icon} {stage_name}: {info['status']} ({info['time']:.1f}с)")

print(f"\nОбщее время выполнения: {total_elapsed/60:.1f} минут")

# Проверка успешности
all_success = all(r["status"] == "success" for r in results.values())

if all_success:
    print("\n" + "=" * 80)
    print("Все эксперименты успешно завершены")
    print("=" * 80)
    print("\nРезультаты сохранены в директории: research/emd_mfdfa/results/")
    print("\nФайлы результатов:")
    print("  - stage2_systematic_results.json/.txt")
    print("  - stage3_exp1_length_effect.json")
    print("  - stage3_exp2_noise_effect.json")
    print("  - stage3_limitations_results.txt")
    print("  - stage4_comparison_results.json/.txt")
else:
    print("\n" + "=" * 80)
    print("Некоторые эксперименты завершились с ошибками")
    print("=" * 80)
    print("\nПроверьте вывод выше для деталей.")
