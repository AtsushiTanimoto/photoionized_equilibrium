"""
Case B 再結合係数の近似式 (HII 領域モデルで recombination_case="B" のとき H, He に使う)
全元素の放射・二電子再結合係数 (Case A) は atomic_data.py が CHIANTI から読む。
"""


def HI_alphaB(T_gas):
    """Case B 水素の再結合係数 [cm^3 s^-1] (Osterbrock & Ferland 2006 の近似式)"""
    return 2.59e-13 * (T_gas / 1e4)**(-0.84)


def HeII_alphaB(T_gas):
    """Case B He II の再結合係数 (He²⁺ + e → He⁺) [cm^3 s^-1]: 水素様スケーリング α_B(He II, T) = 2 α_B(H, T/4)"""
    return 2.0 * HI_alphaB(T_gas / 4.0)


def HeI_alphaB(T_gas):
    """Case B He I の再結合係数 (He⁺ + e → He⁰) [cm^3 s^-1] (Hummer & Storey 1998 に基づく近似式)"""
    return 2.72e-13 * (T_gas / 1e4)**(-0.789)


def HI_alpha2(T_gas):
    """
    水素の n = 2 への再結合係数 α_2s + α_2p [cm^3 s^-1] (Osterbrock & Ferland 2006 の表 2.1、10⁴ K で 0.766×10⁻¹³) の近似式
    (温度依存は α_B と同程度とする)
    """
    return 0.766e-13 * (T_gas / 1e4)**(-0.85)


def HeII_alpha2(T_gas):
    """He II の n = 2 への再結合係数 [cm^3 s^-1]: 水素様スケーリング α_2(He II, T) = 2 α_2(H, T/4)"""
    return 2.0 * HI_alpha2(T_gas / 4.0)


def two_photon_distribution(y):
    """
    水素様イオンの 2s → 1s の 2 光子放射の光子のエネルギー分布 ψ(y) (y = E / E(Lyα))、∫_0^1 ψ dy = 2 (1 回の崩壊で 2 光子) に規格化。
    Nussbaumer & Schmutz (1984) の近似式 A(y) ∝ w [1 − (4w)^γ] + α w^β (4w)^γ (w = y(1−y), α = 0.88, β = 1.53, γ = 0.8)
    """
    import numpy
    y = numpy.asarray(y, dtype=float)
    w = numpy.clip(y * (1.0 - y), 0.0, 0.25)
    A = w * (1.0 - (4.0 * w)**0.8) + 0.88 * w**1.53 * (4.0 * w)**0.8
    grid = numpy.linspace(0.0, 1.0, 2001)
    wg = grid * (1.0 - grid)
    norm = numpy.trapezoid(wg * (1.0 - (4.0 * wg)**0.8) + 0.88 * wg**1.53 * (4.0 * wg)**0.8, grid)
    return 2.0 * A / norm
