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
