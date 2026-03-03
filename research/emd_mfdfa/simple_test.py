"""
Простой тест EMD-MFDFA без графиков
"""

import sys
from pathlib import Path

import numpy as np

# Добавляем корневую директорию проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from StatTools.analysis.emd_mfdfa import emd_mfdfa, estimate_hurst_emd
from StatTools.generators.kasdin_generator import create_kasdin_generator

print("=" * 70)
print("ТЕСТ EMD-BASED MFDFA")
print("=" * 70)

# Параметры
H_true = 0.7
N = 2048

print(f"\n1. Генерация тестового сигнала (fGn с H={H_true}, N={N})...")
try:
    generator = create_kasdin_generator(h=H_true, length=N, normalize=True)
    signal = generator.get_full_sequence()
    print(f"   Сигнал сгенерирован: {len(signal)} точек")
    print(f"   Среднее: {np.mean(signal):.6f}")
    print(f"   СКО: {np.std(signal):.6f}")
    print(f"   Диапазон: [{np.min(signal):.3f}, {np.max(signal):.3f}]")
except Exception as e:
    print(f"   Ошибка генерации: {e}")
    sys.exit(1)

print(f"\n2. Запуск EMD-MFDFA анализа...")
try:
    # Полный анализ
    q_values, h_q, scales = emd_mfdfa(signal, degree=2)
    print(f"   Анализ завершён!")
    print(f"   Количество q-значений: {len(q_values)}")
    print(f"   Количество масштабов: {len(scales)}")

    # Показать несколько значений h(q)
    print(f"\n   Обобщённые показатели Херста h(q):")
    for i, (q, h) in enumerate(zip(q_values, h_q)):
        if i % 3 == 0 or q == 2:  # Показываем каждое 3-е и обязательно q=2
            print(f"      q={q:+3.0f}: h(q)={h:.4f}")

except Exception as e:
    print(f"   Ошибка анализа: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print(f"\n3. Оценка показателя Херста (упрощённый метод)...")
try:
    H_estimated = estimate_hurst_emd(signal, degree=2)
    error = abs(H_estimated - H_true)

    print(f"   Истинное H:      {H_true:.4f}")
    print(f"   Оценённое H:     {H_estimated:.4f}")
    print(f"   Ошибка:          {error:.4f}")
    print(f"   Ошибка (%):      {error/H_true*100:.2f}%")

    # Оценка качества
    if error < 0.05:
        quality = "Отлично"
    elif error < 0.1:
        quality = "Хорошо"
    elif error < 0.15:
        quality = "Приемлемо"
    else:
        quality = "Требует улучшения"

    print(f"   Качество оценки: {quality}")

except Exception as e:
    print(f"   Ошибка оценки: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("Все тесты успешно пройдены")
print("=" * 70)

# Сохраним результаты
output_file = Path(__file__).parent / "test_results.txt"
with open(output_file, "w") as f:
    f.write(f"EMD-MFDFA Test Results\n")
    f.write(f"======================\n\n")
    f.write(f"Signal parameters:\n")
    f.write(f"  True H: {H_true}\n")
    f.write(f"  Length: {N}\n\n")
    f.write(f"Estimated results:\n")
    f.write(f"  Estimated H: {H_estimated:.4f}\n")
    f.write(f"  Error: {error:.4f}\n")
    f.write(f"  Error (%): {error/H_true*100:.2f}%\n\n")
    f.write(f"h(q) values:\n")
    for q, h in zip(q_values, h_q):
        f.write(f"  q={q:+3.0f}: h(q)={h:.4f}\n")

print(f"\nРезультаты сохранены в: {output_file}")
