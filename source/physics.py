"""
物理関数の定義（再結合係数、光電離断面積、プランク関数、ガウント因子）
"""
import numpy
from constants import (h, c, k_B, H01_nu0, H01_a0, He01_nu0, He02_nu0, He01_a0, He02_a0, Lambda_ff_0)


def g_ff(T):
    """
    速度平均した制動放射のガウント因子（Free-Free Gaunt Factor）
    Sutherland (1998, MNRAS, 300, 321) に基づく近似式。
    純水素 (Z=1) の場合に有効。T ~ 10^3 - 10^6 K で g_ff ~ 1.0 - 1.5。
    """
    return 1.1 + 0.34 * numpy.exp(-(5.5 - numpy.log10(T))**2 / 3.0)


# =====================================================================
# 再結合係数 (Case B)
# =====================================================================

def H01_alphaB(T_gas):
    """
    Case B 水素の再結合係数 [cm^3 s^-1]
    Osterbrock & Ferland (2006) の近似式を使用
    """
    return 2.59e-13 * (T_gas / 1e4)**(-0.84)


def He01_alphaB(T_gas):
    """
    Case B He II の再結合係数 (He²⁺ + e → He⁺) [cm^3 s^-1]
    He II は水素様イオン (Z=2) のため、水素のスケーリング則を適用:
      α_B(He II, T) = 2 × α_B(H, T/4)
    """
    return 2.0 * H01_alphaB(T_gas / 4.0)


def He02_alphaB(T_gas):
    """
    Case B He I の再結合係数 (He⁺ + e → He⁰) [cm^3 s^-1]
    Hummer & Storey (1998) に基づく近似式
    """
    return 2.72e-13 * (T_gas / 1e4)**(-0.789)


# =====================================================================
# 光電離断面積
# =====================================================================

def H01_anu(nu):
    """
    水素の光電離断面積 [cm^2]
    nu > nu_0 において (nu_0 / nu)^3 に比例すると近似
    """
    return numpy.where(nu < H01_nu0, 0.0, H01_a0 * (H01_nu0/nu)**3)


def He01_anu(nu):
    """
    He II の光電離断面積 [cm^2]
    水素様イオン (Z=2) なので (ν₀_HeII / ν)^3 に比例
    """
    return numpy.where(nu < He01_nu0, 0.0, He01_a0 * (He01_nu0/ nu)**3)


def He02_anu(nu):
    """
    He I の光電離断面積 [cm^2]
    Osterbrock & Ferland (2006) の近似式:
      σ(ν) = a_0_HeI × [1.66 (ν₁/ν)^2.05 − 0.66 (ν₁/ν)^3.05]
    閾値 (ν = ν₁) で σ = a_0_HeI、高振動数で ≈ ν^-2 に漸近。
    """
    x = He02_nu0 / nu
    sigma = He02_a0 * (1.66 * x**2.05 - 0.66 * x**3.05)
    return numpy.where(nu < He02_nu0, 0.0, sigma)


# =====================================================================
# 黒体放射
# =====================================================================
def B_nu(nu, T_BB):
    """
    プランク関数（黒体放射） [erg s^-1 cm^-2 Hz^-1 sr^-1]
    """
    exponent = h * nu / (k_B * T_BB)
    exponent = numpy.minimum(exponent, 700.0) # オーバーフロー対策
    return (2 * h * nu**3 / c**2) / (numpy.exp(exponent) - 1.0)
