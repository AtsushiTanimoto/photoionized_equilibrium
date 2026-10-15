"""
H 原子・H⁺ との電荷交換の係数 (Kingdon & Ferland 1996, ApJS, 106, 205 の解析フィット)

  再結合側: A^(q+1)+ + H  → A^q+ + H⁺   k_rec(T)  (親イオンの電荷 q+1 = 1〜4)
  電離側:   A^q+ + H⁺     → A^(q+1)+ + H k_ion(T)  (親イオンの電荷 q = 0〜3)

データは Cloudy の ctrecombdata.dat / ctiondata.dat (constants.CHARGE_EXCHANGE_DIR、書式は同じディレクトリの README.md)。
式: k = a 10⁻⁹ (T₄)^b [1 + c exp(d T₄)]、T₄ = T/10⁴ K を [T_low, T_high] に切り詰める。
電離側はさらに exp(−ΔT/T) の Boltzmann 因子が掛かる (ΔT は切り詰めない温度で評価)。
データのない組 (a = 0) と電荷 4 以上 (電離側は 3 以上) は 0。
"""
import os

import numpy

import constants


MAX_PARENT_CHARGE_RECOMBINATION = 4   # 再結合側の親イオンの電荷の上限 (データは 1〜4)
MAX_PARENT_CHARGE_IONIZATION = 3      # 電離側の親イオンの電荷の上限 (データは 0〜3)

_recombination_data = None
_ionization_data = None


def _read_table(filename, N_columns):
    """Cloudy の電荷交換データファイルを読む。戻り値: {(Z, 行番号 0〜3): [係数]}"""
    table = {}
    with open(os.path.join(constants.CHARGE_EXCHANGE_DIR, filename)) as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    rows = lines[1:]   # 1 行目はマジックナンバー
    for n, line in enumerate(rows):
        columns = [float(v) for v in line.split()]
        if len(columns) != N_columns:
            raise ValueError(f"{filename}: {n + 2} 行目の列数が {len(columns)} (期待 {N_columns})")
        Z, row = n // 4 + 1, n % 4
        table[(Z, row)] = columns
    return table


def _load():
    global _recombination_data, _ionization_data
    if _recombination_data is None:
        _recombination_data = _read_table("ctrecombdata.dat", 7)
        _ionization_data = _read_table("ctiondata.dat", 8)


def _fit(T_gas, a, b, c, d, T_low, T_high):
    T_used = numpy.clip(T_gas, T_low, T_high) * 1.0e-4
    return a * 1.0e-9 * T_used**b * (1.0 + c * numpy.exp(d * T_used))


def recombination_rate(Z, parent_charge, T_gas):
    """
    A^(parent_charge)+ + H → A^(parent_charge−1)+ + H⁺ の係数 [cm³ s⁻¹]。T_gas は配列でもよい。
    データがなければ 0。
    """
    _load()
    T_gas = numpy.asarray(T_gas, dtype=float)
    if parent_charge < 1 or parent_charge > MAX_PARENT_CHARGE_RECOMBINATION:
        return numpy.zeros_like(T_gas)
    a, b, c, d, T_low, T_high, _ = _recombination_data.get((Z, parent_charge - 1), [0.0] * 7)
    if a <= 0.0:
        return numpy.zeros_like(T_gas)
    return _fit(T_gas, a, b, c, d, T_low, T_high)


def ionization_rate(Z, parent_charge, T_gas):
    """
    A^(parent_charge)+ + H⁺ → A^(parent_charge+1)+ + H の係数 [cm³ s⁻¹]。T_gas は配列でもよい。
    データがなければ 0。
    """
    _load()
    T_gas = numpy.asarray(T_gas, dtype=float)
    if parent_charge < 0 or parent_charge > MAX_PARENT_CHARGE_IONIZATION:
        return numpy.zeros_like(T_gas)
    a, b, c, d, T_low, T_high, delta_T4, _ = _ionization_data.get((Z, parent_charge), [0.0] * 8)
    if a <= 0.0:
        return numpy.zeros_like(T_gas)
    return _fit(T_gas, a, b, c, d, T_low, T_high) * numpy.exp(-delta_T4 * 1.0e4 / numpy.maximum(T_gas, 1.0))


def energy_defect_eV(Z, parent_charge, direction):
    """反応のエネルギー差 [eV] (direction = "recombination" | "ionization"。将来の加熱・冷却用)"""
    _load()
    if direction == "recombination":
        return _recombination_data.get((Z, parent_charge - 1), [0.0] * 7)[6]
    return _ionization_data.get((Z, parent_charge), [0.0] * 8)[7]
