# EMD-based MFDFA Research

Исследование метода EMD-based MFDFA (Empirical Mode Decomposition combined with Multifractal Detrended Fluctuation Analysis) для анализа флуктуационных характеристик временных рядов.

## Описание метода

EMD-based MFDFA представляет собой гибридный метод, объединяющий:

1. EMD (Empirical Mode Decomposition) — адаптивное разложение сигнала на внутренние модовые функции (IMF)
2. MFDFA (Multifractal Detrended Fluctuation Analysis) — анализ мультифрактальных свойств

Данный подход обеспечивает более точный анализ нестационарных сигналов со сложными трендами по сравнению с классическими методами.

## Структура проекта

```
emd_mfdfa/
├── README.md                           # Описание проекта
├── __init__.py                         # Инициализация модуля
│
├── generate_test_data.py               # Генерация синтетических данных
├── simple_test.py                      # Простой тест метода
│
├── experiment_stage2_systematic.py     # Систематические эксперименты
├── experiment_stage3_limitations.py    # Исследование ограничений
├── experiment_stage4_comparison.py     # Сравнение с DFA
├── run_all_experiments.py              # Запуск всех экспериментов
│
└── results/                            # Результаты экспериментов
    ├── stage2_systematic_results.json
    ├── stage2_systematic_results.txt
    ├── stage3_exp1_length_effect.json
    ├── stage3_exp2_noise_effect.json
    ├── stage3_limitations_results.txt
    ├── stage4_comparison_results.json
    └── stage4_comparison_results.txt
```

## Установка зависимостей

Для работы с EMD требуется библиотека PyEMD:

```bash
pip install EMD-signal
```

Остальные зависимости уже установлены в основном проекте.

## Использование

### 1. Простой тест

```bash
cd research/emd_mfdfa
python simple_test.py
```

Выполняет базовый тест метода на синтетических данных с H=0.7.

### 2. Запуск всех экспериментов

```bash
python run_all_experiments.py
```

Последовательно выполняет все этапы экспериментального исследования (около 3-5 минут).

### 3. Отдельные этапы экспериментов

```bash
# Систематические тесты (80 экспериментов)
python experiment_stage2_systematic.py

# Исследование ограничений (50 экспериментов)
python experiment_stage3_limitations.py

# Сравнение с DFA (80 экспериментов)
python experiment_stage4_comparison.py
```

### 4. Использование в коде

```python
from StatTools.analysis.emd_mfdfa import emd_mfdfa, estimate_hurst_emd
from StatTools.generators.kasdin_generator import create_kasdin_generator

# Генерация тестового сигнала
generator = create_kasdin_generator(h=0.7, length=2**12)
signal = generator.get_full_sequence()

# Анализ
q_values, h_q, scales = emd_mfdfa(signal, degree=2)

# Или просто оценка показателя Херста
H = estimate_hurst_emd(signal)
print(f"Hurst exponent: {H:.3f}")
```

## План эксперимента

### Этап 1: Верификация
- Реализация функции `emd_mfdfa()`
- Генератор синтетических данных (fGn)
- Базовые тесты функциональности

### Этап 2: Основной эксперимент
- Полная реализация MFDFA-ядра
- Систематические эксперименты на синтетических данных (80 реализаций)
- Оценка точности: MAE = 0.0351 (средняя абсолютная ошибка 3.51%)

### Этап 3: Исследование ограничений
- Влияние длины ряда (N): оптимальные результаты при N ≥ 4096
- Влияние уровня шума (SNR): метод устойчив к шуму 20-30 dB
- Результат: точность возрастает с увеличением длины ряда

### Этап 4: Сравнительный анализ
- Сравнение с классическим методом DFA
- Анализ преимуществ гибридного подхода
- Результат: EMD-MFDFA превосходит DFA по точности на 34% (MAE: 0.0295 vs 0.0448)

## Параметры эксперимента

| Параметр | Значения | Обоснование |
|----------|----------|-------------|
| H (Hurst exponent) | 0.2, 0.3, ..., 0.9 | Охват диапазона от антиперсистентных до персистентных процессов |
| N (длина ряда) | 512, 1024, 2048, 4096, 8192 | Исследование влияния длины временного ряда |
| SNR (отношение сигнал/шум) | ∞, 30, 20, 10, 5 dB | Робастность метода к шуму |
| m (степень полинома) | 1, 2 | Порядок детрендинга в MFDFA |

## Метрики качества

- Точность: MAE (Mean Absolute Error) = |H_estimated - H_true|
- Устойчивость: стандартное отклонение σ(H) при повторных реализациях
- Производительность: среднее время выполнения анализа

## Библиография

1. Huang, N. E., Shen, Z., Long, S. R., Wu, M. C., Shih, H. H., Zheng, Q., Yen, N.-C., Tung, C. C., & Liu, H. H. (1998). The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis. *Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences*, 454(1971), 903-995.
   DOI: [10.1098/rspa.1998.0193](https://doi.org/10.1098/rspa.1998.0193)

2. Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E., Havlin, S., Bunde, A., & Stanley, H. E. (2002). Multifractal detrended fluctuation analysis of nonstationary time series. *Physica A: Statistical Mechanics and its Applications*, 316(1-4), 87-114.
   DOI: [10.1016/S0378-4371(02)01383-3](https://doi.org/10.1016/S0378-4371(02)01383-3)

3. Xi-Yuan Qian, Gao-Feng Gu, Wei-Xing Zhou (2011). Modified detrended fluctuation analysis based on empirical mode decomposition for the characterization of anti-persistent processes. *Physica A: Statistical Mechanics and its Applications*, 390(23–24), 4388-4395.
DOI: [10.1016/j.physa.2011.07.008](https://doi.org/10.1016/j.physa.2011.07.008)
