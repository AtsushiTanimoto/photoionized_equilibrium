"""
制動放射（自由-自由放射）に関する関数

全周波数で積分し熱平均した Gaunt 因子 ⟨g_ff⟩(γ²) は van Hoof et al. (2015, MNRAS, 449, 2112) の表
(constants.GAUNT_FF_DIR/gauntff_freqint_Z<ZZ>.dat、相対論補正込み) を log₁₀ γ² で 3 次スプライン内挿する。
γ² = Z² Ry / kT で、Z はイオンの電荷 (水素様の近似)。
"""
import os

import astropy.constants
import astropy.units
import numpy
import scipy.interpolate

import constants


Ry_over_k = (astropy.constants.Ryd * astropy.constants.h * astropy.constants.c / astropy.constants.k_B).to(astropy.units.K).value   # 157887.5 K

_splines = {}


def _load(Z):
    """電荷 Z の表を読み、log₁₀ γ² → ⟨g_ff⟩ のスプラインを返す"""
    if Z not in _splines:
        path = os.path.join(constants.GAUNT_FF_DIR, f"gauntff_freqint_Z{Z:02d}.dat")
        header, rows = [], []
        with open(path) as f:
            for line in f:
                text = line.split("#")[0].split()
                if not text:
                    continue
                if len(header) < 5:
                    header.append(float(text[0]))   # magic number, 格子点数, log γ² の開始値, 刻み, Z
                else:
                    rows.append([float(v) for v in text])
        N, start, step = int(header[1]), header[2], header[3]
        table = numpy.array(rows)
        if not numpy.allclose(table[:, 0], start + step * numpy.arange(N)):
            raise ValueError(f"{path}: log(gam2) の格子が見出しと合わない")
        _splines[Z] = scipy.interpolate.CubicSpline(table[:, 0], table[:, 1])
    return _splines[Z]


def total_gaunt_factor(Z, T):
    """
    電荷 Z (≥ 1) のイオンの、周波数積分・熱平均した制動放射の Gaunt 因子 ⟨g_ff⟩。T [K] は配列でもよい。
    表の範囲 (log₁₀ γ² = −6〜10) の外は端の値に切り詰める。
    """
    T = numpy.asarray(T, dtype=float)
    log_gamma2 = numpy.clip(numpy.log10(Z**2 * Ry_over_k / T), -6.0, 10.0)
    return _load(int(Z))(log_gamma2)
