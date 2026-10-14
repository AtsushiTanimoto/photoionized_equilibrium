"""
制動放射（自由-自由放射）に関する関数
"""
import numpy


def g_ff(T):
    """
    速度平均した制動放射のガウント因子（Free-Free Gaunt Factor）
    Sutherland (1998, MNRAS, 300, 321) に基づく近似式。
    純水素 (Z=1) の場合に有効。T ~ 10^3 - 10^6 K で g_ff ~ 1.0 - 1.5。
    """
    return 1.1 + 0.34 * numpy.exp(-(5.5 - numpy.log10(T))**2 / 3.0)
