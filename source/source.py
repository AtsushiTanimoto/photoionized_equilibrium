"""
光源のスペクトル
  BlackbodySource: 黒体（星の半径で規格化、または XSTAR の bbody と同様に光度で規格化）
  TabulatedSource: 表で与えたスペクトル（XSTAR の xout_spect1.fits の incident など）
いずれも luminosity_density(E) [erg s^-1 eV^-1] を与え、エネルギービンごとの光子放出率と
モンテカルロのパケットのサンプリングを共通に扱う。
"""
import numpy

from constants import h, c, k_B, eV, R_sun


class Source:
    """光源の基底クラス。サブクラスは luminosity_density(E_eV) [erg s^-1 eV^-1] を実装する。"""
    E_min_eV = 1.0        # スペクトルを定義する範囲（サブクラスで上書き）
    E_max_eV = 1.0e6

    def luminosity_density(self, E_eV):
        raise NotImplementedError

    def photon_rate_density(self, E_eV):
        """光子放出率のスペクトル [photons s^-1 eV^-1]"""
        E_eV = numpy.asarray(E_eV, dtype=float)
        return self.luminosity_density(E_eV) / (E_eV * eV)

    def photon_rate_in_band(self, E_lo_eV, E_hi_eV, N=20001):
        """[E_lo, E_hi] の光子放出率 [s^-1]（対数グリッドで台形積分）"""
        E = numpy.geomspace(E_lo_eV, E_hi_eV, N)
        return numpy.trapezoid(self.photon_rate_density(E), E)

    def luminosity_in_band(self, E_lo_eV, E_hi_eV, N=20001):
        """[E_lo, E_hi] の光度 [erg s^-1]"""
        E = numpy.geomspace(E_lo_eV, E_hi_eV, N)
        return numpy.trapezoid(self.luminosity_density(E), E)

    def photon_rate_per_bin(self, E_edges_eV, N_sub=16):
        """
        エネルギービン（境界 E_edges_eV）ごとの光子放出率 [s^-1]
        各ビンを N_sub 分割した対数グリッドで台形積分する。
        """
        E_edges_eV = numpy.asarray(E_edges_eV, dtype=float)
        log_edges = numpy.log(E_edges_eV)
        Q = numpy.zeros(len(E_edges_eV) - 1)
        for k in range(len(Q)):
            E = numpy.exp(numpy.linspace(log_edges[k], log_edges[k + 1], N_sub + 1))
            Q[k] = numpy.trapezoid(self.photon_rate_density(E), E)
        return Q

    def sample_packet_counts(self, Q_bin, N_packets, rng):
        """
        光子パケットをビンごとの放出率に比例してサンプリングし、ビンごとのパケット数を返す
        （パケットのエネルギーをサンプリングしてヒストグラムにするのと同じ）。
        """
        p = Q_bin / numpy.sum(Q_bin)
        return rng.multinomial(N_packets, p).astype(float)


class BlackbodySource(Source):
    """
    黒体放射の光源
      radius_cm を与えると L_E = 4π R^2 π B_E（星の表面からの放射）
      luminosity を与えると、band [eV] 内の光度が luminosity [erg s^-1] になるように規格化する
      （XSTAR の spectrum=bbody, rlrad38 と同じ規格化: 既定の band は 1-1000 Ry）
    """
    def __init__(self, T_BB, radius_cm=None, luminosity=None, band_eV=(13.6, 13600.0)):
        self.T_BB = T_BB
        self.kT_eV = k_B * T_BB / eV
        self.E_min_eV = 1.0e-3 * self.kT_eV
        self.E_max_eV = 40.0 * self.kT_eV   # exp(-40) まで
        if radius_cm is None and luminosity is None:
            raise ValueError("Give either radius_cm or luminosity")
        self.radius_cm = radius_cm
        self.luminosity = luminosity
        self.band_eV = band_eV
        self._normalization = 1.0
        if luminosity is not None:
            self._normalization = luminosity / self.luminosity_in_band(*band_eV)
        else:
            self._normalization = 4.0 * numpy.pi * radius_cm**2 * numpy.pi

    def luminosity_density(self, E_eV):
        """L_E [erg s^-1 eV^-1]: B_E = 2 E^3 / (h^3 c^2) / (exp(E/kT) - 1) に規格化定数をかける（E は eV）"""
        E_eV = numpy.asarray(E_eV, dtype=float)
        x = numpy.minimum(E_eV / self.kT_eV, 700.0)
        B_E = 2.0 * (E_eV * eV)**3 / (h**3 * c**2) / numpy.expm1(x) * eV  # [erg s^-1 cm^-2 sr^-1 eV^-1]
        return self._normalization * B_E


class TabulatedSource(Source):
    """表で与えた光源スペクトル (E [eV] 昇順、L_E [erg s^-1 eV^-1])。範囲外は 0。対数-対数で線形内挿する。"""
    def __init__(self, E_eV, L_E):
        E_eV = numpy.asarray(E_eV, dtype=float)
        L_E = numpy.asarray(L_E, dtype=float)
        positive = L_E > 0.0
        self.E_table = E_eV[positive]
        self.log_E = numpy.log(self.E_table)
        self.log_L = numpy.log(L_E[positive])
        self.E_min_eV = self.E_table[0]
        self.E_max_eV = self.E_table[-1]

    def luminosity_density(self, E_eV):
        E_eV = numpy.asarray(E_eV, dtype=float)
        inside = (E_eV >= self.E_min_eV) & (E_eV <= self.E_max_eV)
        L = numpy.zeros_like(E_eV)
        L[inside] = numpy.exp(numpy.interp(numpy.log(E_eV[inside]), self.log_E, self.log_L))
        return L

    @classmethod
    def from_xstar(cls, filepath, column="incident"):
        """XSTAR の xout_spect1.fits (XSTAR_SPECTRA: energy [eV], incident [10^38 erg s^-1 erg^-1]) から作る"""
        import astropy.io.fits
        data = astropy.io.fits.open(filepath)["XSTAR_SPECTRA"].data
        E_eV = data["energy"].astype(float)
        L_E = data[column].astype(float) * 1.0e38 * eV   # erg s^-1 erg^-1 → erg s^-1 eV^-1
        return cls(E_eV, L_E)
