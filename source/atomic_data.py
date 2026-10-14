"""
原子データの読み込み (全元素・全電離段階で共通)

  - 光電離断面積: FAC 原子データベース (<ATOMIC_DATA_DIR>/<元素>/<元素><電子数>.pi)。
    MONACO と同じ式 σ(E) = Σ_k Σ0_k (E/I_k)^Γ_k exp(-E/τ_k) で評価する。
    データベースには電子数 ≤ 10 (H, He, C, N, O, Ne は全段階) のイオンしかないため、
    それより低い電離段階は扱わない (ElementData.min_charge を参照)。
  - 電離ポテンシャル: CHIANTI の ip/chianti.ip (実験値)。FAC の閾値はこれに合わせてずらす。
  - 再結合係数: CHIANTI の *.rrparams (Badnell 2006 放射再結合) と *.drparams (二電子再結合のフィット)。
  - 衝突電離: CHIANTI の *.diparams (Dere 2007 のスケールした断面積) を Maxwell 分布で積分する。

イオンは電荷 j (0: 中性, ..., Z: 裸) で指定する。CHIANTI のイオン番号 (分光学的記法) は j+1、
MONACO/XSTAR の CSV のイオン記号は <元素><電子数 2 桁> (例: Fe24 = Fe III? ではなく 24 電子 = Fe III)。
"""
import numpy
import os
import scipy.interpolate

from constants import h, k_B, eV, m_e, ATOMIC_DATA_DIR, CHIANTI_DIR, ATOMIC_NUMBER


CM_INVERSE_TO_EV = 1.239841984e-4  # 1 cm^-1 = 1.2398e-4 eV
Ry = 2.1798723611e-11               # リュードベリ [erg]


# ---------------------------------------------------------------------------
# CHIANTI のファイル読み込み
# ---------------------------------------------------------------------------
def read_chianti_numeric_lines(filepath):
    """CHIANTI のファイルの数値部分（最初の "-1" の行まで）を行ごとに返す"""
    lines = []
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if stripped == "-1" or stripped.startswith("%"):
                break
            try:
                float(stripped.split()[0])
            except (ValueError, IndexError):
                break  # 終端の "-1" がないファイルでは、数値でない行で終わる
            lines.append(line)
    return lines


def _chianti_ion_directory(element, ion):
    name = f"{element.lower()}_{ion}"
    return os.path.join(CHIANTI_DIR, element.lower(), name), name


_ionization_potentials = None


def ionization_potential(Z, charge):
    """
    電荷 charge のイオンの電離ポテンシャル [eV] (CHIANTI の ip/chianti.ip)。
    """
    global _ionization_potentials
    if _ionization_potentials is None:
        _ionization_potentials = {}
        with open(os.path.join(CHIANTI_DIR, "ip", "chianti.ip")) as f:
            for line in f:
                columns = line.split()
                if len(columns) < 3 or line.strip() == "-1":
                    break
                _ionization_potentials[(int(columns[0]), int(columns[1]))] = float(columns[2]) * CM_INVERSE_TO_EV
    return _ionization_potentials[(Z, charge + 1)]


def load_chianti_recombination(element, charge):
    """
    電荷 charge のイオン (CHIANTI のイオン番号 charge+1) から 1 階下への再結合係数のフィットパラメータ。
    戻り値: (rr_type, rr_parameters, dr_energies [K], dr_coefficients)
      二電子再結合のデータがない場合（水素様 → 中性など）は dr_* は空の配列
    """
    directory, name = _chianti_ion_directory(element, charge + 1)
    rr_rows = [[float(v) for v in line.split()] for line in read_chianti_numeric_lines(os.path.join(directory, f"{name}.rrparams"))]
    rr_type = int(rr_rows[0][0])
    if rr_type in (1, 2):
        rr_parameters = rr_rows[1][3:]  # 先頭の Z, ion, 準位番号 を除く
    elif rr_type == 3:
        rr_parameters = rr_rows[1][2:]  # 先頭の Z, ion を除く (A, η)
    else:
        raise ValueError(f"Unsupported RR fit type {rr_type} in {name}.rrparams")
    dr_filepath = os.path.join(directory, f"{name}.drparams")
    if os.path.exists(dr_filepath):
        dr_rows = [[float(v) for v in line.split()] for line in read_chianti_numeric_lines(dr_filepath)]
        dr_type = int(dr_rows[0][0])
        if dr_type == 1:
            dr_energies = numpy.array(dr_rows[1][2:])
            dr_coefficients = numpy.array(dr_rows[2][2:])
        elif dr_type == 2:
            # Shull & Van Steenberg (1982) 形式 (A, B, T0, T1): α_DR = A T^-3/2 exp(-T0/T) [1 + B exp(-T1/T)]
            # = T^-3/2 [A exp(-T0/T) + A B exp(-(T0+T1)/T)] なので type 1 と同じ 2 項の形に直す
            A, B, T0, T1 = dr_rows[1][2:6]
            dr_energies = numpy.array([T0, T0 + T1])
            dr_coefficients = numpy.array([A, A * B])
        else:
            raise ValueError(f"Unsupported DR fit type {dr_type} in {dr_filepath}")
    else:
        dr_energies = numpy.zeros(0)
        dr_coefficients = numpy.zeros(0)
    return rr_type, rr_parameters, dr_energies, dr_coefficients


def radiative_recombination_rate(T_gas, rr_type, rr_parameters):
    """
    放射再結合係数 [cm^3 s^-1] (Badnell 2006, ApJS 167, 334 のフィット式)
      α_RR = A / [√(T/T0) (1 + √(T/T0))^(1-B) (1 + √(T/T1))^(1+B)]
      rr_type = 2 では B → B + C exp(-T2/T)
      rr_type = 3 は Shull & Van Steenberg (1982) のべき乗則 α_RR = A (T/10^4 K)^-η
    """
    T_gas = numpy.asarray(T_gas, dtype=float)
    if rr_type == 3:
        A, eta = rr_parameters
        return A * (T_gas / 1.0e4)**(-eta)
    if rr_type == 1:
        A, B, T0, T1 = rr_parameters
    elif rr_type == 2:
        A, B, T0, T1, C, T2 = rr_parameters
        B = B + C * numpy.exp(-T2 / T_gas)
    else:
        raise ValueError(f"Unsupported RR fit type {rr_type}")
    x0 = numpy.sqrt(T_gas / T0)
    x1 = numpy.sqrt(T_gas / T1)
    return A / (x0 * (1.0 + x0)**(1.0 - B) * (1.0 + x1)**(1.0 + B))


T_HI = 157807.0  # 水素の電離ポテンシャル / k_B [K]


def radiative_recombination_energy_factor(T_gas, charge):
    """
    放射再結合で持ち去られる電子の平均運動エネルギー ⟨E⟩ を kT で割った値。
    水素様イオン (電荷 charge のイオンへの再結合) の Case A の値 (Hui & Gnedin 1997 のフィット β_A/(α_A kT)) を
    温度を T/charge^2 にスケールして使う（低温で ≈ 0.7、高温 T ≫ charge^2 T_HI で ≈ 0.3）。
    非水素様イオンにもこの近似を用いる。
    """
    T_gas = numpy.asarray(T_gas, dtype=float)
    z = max(charge, 1)
    lam = 2.0 * z**2 * T_HI / T_gas
    alpha = 1.269e-13 * lam**1.503 / (1.0 + (lam / 0.522)**0.47)**1.923
    beta = 1.778e-29 * (T_gas / z**2) * lam**1.965 / (1.0 + (lam / 0.541)**0.502)**2.697
    return beta / (alpha * k_B * T_gas / z**2)


def dielectronic_recombination_rate(T_gas, dr_energies, dr_coefficients):
    """
    二電子再結合係数 [cm^3 s^-1] (Badnell らのフィット式)
      α_DR = T^(-3/2) Σ_i c_i exp(-E_i/T)
    """
    T_gas = numpy.asarray(T_gas, dtype=float)[..., None]
    return numpy.sum(dr_coefficients * numpy.exp(-dr_energies / T_gas), axis=-1) * numpy.squeeze(T_gas, axis=-1)**-1.5


def dielectronic_recombination_cooling(T_gas, dr_energies, dr_coefficients):
    """
    二電子再結合による冷却係数 [erg cm^3 s^-1]
    各項の共鳴エネルギー k_B E_i を捕獲される電子が持ち去るとみなす: Λ_DR/(n_e n) = T^(-3/2) Σ_i c_i k_B E_i exp(-E_i/T)
    """
    T_gas = numpy.asarray(T_gas, dtype=float)[..., None]
    return numpy.sum(dr_coefficients * k_B * dr_energies * numpy.exp(-dr_energies / T_gas), axis=-1) * numpy.squeeze(T_gas, axis=-1)**-1.5


def load_chianti_direct_ionization(element, charge):
    """
    電荷 charge のイオンの直接衝突電離断面積のフィット (CHIANTI の *.diparams, Dere 2007)。
    形式: 1 行目 Z, イオン番号, 節点数, 殻の数 nfac, 励起自動電離の有無 neaev
          殻ごとに 2 行: (f, スケールした エネルギー x の節点...), (電離閾値 I [eV], スケールした断面積 y の節点...)
    戻り値: [(f, I [eV], x_knots, y_knots), ...] (殻ごと)。データがなければ空のリスト。
    励起自動電離 (*.easplups) は未実装。
    """
    directory, name = _chianti_ion_directory(element, charge + 1)
    filepath = os.path.join(directory, f"{name}.diparams")
    if not os.path.exists(filepath):
        return []
    lines = read_chianti_numeric_lines(filepath)
    header = lines[0].split()
    nfac = int(header[3])
    shells = []
    for k in range(nfac):
        row_x = [float(v) for v in lines[1 + 2 * k].split()]
        row_y = [float(v) for v in lines[2 + 2 * k].split()]
        shells.append((row_x[0], row_y[0], numpy.array(row_x[1:]), numpy.array(row_y[1:])))
    return shells


def direct_ionization_cross_section(E_eV, shells):
    """
    衝突電離断面積 [cm^2] (Dere 2007 のスケーリングを戻す)
      u = E/I, x = 1 - ln f / ln(u - 1 + f), σ = y(x) ln(u - 1 + f) / (u I^2) × 10^-14 cm^2 (I は eV)
    """
    E_eV = numpy.asarray(E_eV, dtype=float)
    sigma = numpy.zeros_like(E_eV)
    for f, I, x_knots, y_knots in shells:
        u = E_eV / I
        above = u > 1.0
        if not numpy.any(above):
            continue
        u_above = u[above]
        x = 1.0 - numpy.log(f) / numpy.log(u_above - 1.0 + f)
        spline = scipy.interpolate.splrep(x_knots, y_knots, s=0, k=min(3, len(x_knots) - 1))
        y = numpy.maximum(scipy.interpolate.splev(numpy.clip(x, 0.0, 1.0), spline), 0.0)
        sigma[above] += y * numpy.log(u_above - 1.0 + f) / (u_above * I**2) * 1.0e-14
    return sigma


def direct_ionization_rate(T_gas, shells, N_u=400):
    """
    衝突電離係数 q(T) [cm^3 s^-1]: Maxwell 分布で断面積を積分する
      q = ∫ σ(E) v f(E) dE,  f(E) = 2 √(E/π) (kT)^(-3/2) exp(-E/kT),  v = √(2E/m_e)
    T_gas は配列でもよい（温度ごとにベクトル化）。
    """
    if not shells:
        return numpy.zeros_like(numpy.asarray(T_gas, dtype=float))
    T_gas = numpy.asarray(T_gas, dtype=float)
    I_min = min(I for _, I, _, _ in shells)
    # 閾値から 10^4 倍まで対数グリッド（低温では exp(-E/kT) が急減するので十分）
    u = numpy.geomspace(1.0, 1.0e4, N_u)
    E = I_min * u                                       # [eV]
    sigma = direct_ionization_cross_section(E, shells)  # [cm^2]
    v = numpy.sqrt(2.0 * E * eV / m_e)                  # [cm s^-1]
    kT = k_B * T_gas[..., None] / eV                     # [eV]
    f_E = 2.0 * numpy.sqrt(E / numpy.pi) * kT**-1.5 * numpy.exp(-numpy.minimum(E / kT, 700.0))  # [eV^-1]
    return numpy.trapezoid(sigma * v * f_E, E, axis=-1)


# ---------------------------------------------------------------------------
# FAC のファイル読み込み
# ---------------------------------------------------------------------------
def fac_ion_symbol(element, number_of_electrons):
    """MONACO/XSTAR CSV のイオン記号 (元素記号 + 電子数 2 桁)"""
    return f"{element}{number_of_electrons:02d}"


def load_photoionization_channels(element, number_of_electrons, ionization_potential_eV):
    """
    FAC 原子データベース (<ATOMIC_DATA_DIR>/<元素>/<元素><電子数>.pi) から
    基底準位の光電離断面積のフィットパラメータを読み込む。
    各行: 束縛準位, 2J, 電離後の準位, 2J, 軌道 (n*1000+l), 電離ポテンシャル I [eV], Σ0 [cm^2], Γ, τ [eV]
    多電子イオンでは基底準位から複数のチャネル（電離後の励起準位や内殻電離）があるため、
    基底準位の行をすべて読み込む。
    FAC の電離ポテンシャルは実験値からずれている（例: C I で 10.317 eV、実験値 11.260 eV）ため、
    最低閾値のチャネルが実験値 ionization_potential_eV に一致するよう、全チャネルの I を同じ量だけずらす
    （チャネル間の閾値の差は FAC の値を保つ）。
    戻り値: [(I [eV], Σ0 [cm^2], Γ, τ [eV]), ...] (電離ポテンシャルの昇順)。ファイルがなければ None。
    """
    filepath = os.path.join(ATOMIC_DATA_DIR, element, f"{fac_ion_symbol(element, number_of_electrons)}.pi")
    if not os.path.exists(filepath):
        return None
    channels = []
    with open(filepath) as f:
        for line in f:
            columns = line.split()
            if columns and int(columns[0]) == 0:  # 基底準位からの光電離
                channels.append((float(columns[5]), float(columns[6]), float(columns[7]), float(columns[8])))
    if not channels:
        raise ValueError(f"No photoionization data for the ground level in {filepath}")
    channels.sort()
    shift = ionization_potential_eV - channels[0][0]
    return [(I + shift, Sigma0, Gamma, tau) for I, Sigma0, Gamma, tau in channels]


def photoionization_cross_section(E_eV, channels):
    """
    光電離断面積 [cm^2] (MONACO の PhotoionizationData::CrossSection と同じ式)
      σ(E) = Σ_k Σ0_k (E/I_k)^Γ_k exp(-E/τ_k)  (各チャネル k について E ≥ I_k の項のみ)
    """
    E = numpy.asarray(E_eV, dtype=float)
    sigma = numpy.zeros_like(E)
    for I, Sigma0, Gamma, tau in channels:
        sigma += numpy.where(E < I, 0.0, Sigma0 * (E / I)**Gamma * numpy.exp(-E / tau))
    return sigma


def photoionization_heating_cross_section(E_eV, channels):
    """
    光電離加熱の断面積 Σ_k σ_k(E) (E - I_k) [erg cm^2]
    各チャネルで光電子が持ち去るエネルギー E - I_k が加熱に寄与する
    （電離後のイオンの励起エネルギーは放射で失われるとみなす。内殻電離に続く Auger 電離は未実装）
    """
    E = numpy.asarray(E_eV, dtype=float)
    heating = numpy.zeros_like(E)
    for I, Sigma0, Gamma, tau in channels:
        heating += numpy.where(E < I, 0.0, Sigma0 * (E / I)**Gamma * numpy.exp(-E / tau) * (E - I) * eV)
    return heating


# ---------------------------------------------------------------------------
# イオン・元素のデータをまとめるクラス
# ---------------------------------------------------------------------------
class IonData:
    """
    元素 element の電荷 charge のイオンの原子データ。
      channels:   光電離断面積のチャネル (裸のイオンや FAC データがないイオンでは None)
      ip_eV:      電離ポテンシャル [eV] (裸のイオンでは None)
      rr, dr:     このイオンから 1 階下への再結合 (中性では None)
      di_shells:  このイオンの衝突電離断面積 (裸のイオンでは空)
    """
    def __init__(self, element, charge):
        self.element = element
        self.Z = ATOMIC_NUMBER[element]
        self.charge = charge
        self.number_of_electrons = self.Z - charge
        self.symbol = fac_ion_symbol(element, self.number_of_electrons)  # 例: Fe24, 裸は Fe00
        self.chianti_name = f"{element.lower()}_{charge + 1}"
        if charge < self.Z:
            self.ip_eV = ionization_potential(self.Z, charge)
            self.channels = load_photoionization_channels(element, self.number_of_electrons, self.ip_eV)
            self.di_shells = load_chianti_direct_ionization(element, charge)
        else:
            self.ip_eV = None
            self.channels = None
            self.di_shells = []
        if charge > 0:
            self.rr_type, self.rr_parameters, self.dr_energies, self.dr_coefficients = load_chianti_recombination(element, charge)
        else:
            self.rr_type = None

    @property
    def has_photoionization_data(self):
        return self.channels is not None

    def photoionization_cross_section(self, E_eV):
        """光電離断面積 [cm^2]（データがなければ 0）"""
        if self.channels is None:
            return numpy.zeros_like(numpy.asarray(E_eV, dtype=float))
        return photoionization_cross_section(E_eV, self.channels)

    def photoionization_heating_cross_section(self, E_eV):
        """光電離加熱の断面積 [erg cm^2]"""
        if self.channels is None:
            return numpy.zeros_like(numpy.asarray(E_eV, dtype=float))
        return photoionization_heating_cross_section(E_eV, self.channels)

    def alpha_RR(self, T_gas):
        """放射再結合係数 (このイオン → 1 階下) [cm^3 s^-1]"""
        if self.rr_type is None:
            return numpy.zeros_like(numpy.asarray(T_gas, dtype=float))
        return radiative_recombination_rate(T_gas, self.rr_type, self.rr_parameters)

    def rr_cooling(self, T_gas):
        """放射再結合による冷却係数 [erg cm^3 s^-1] = α_RR ⟨E⟩, ⟨E⟩ = kT × radiative_recombination_energy_factor"""
        T_gas = numpy.asarray(T_gas, dtype=float)
        if self.rr_type is None:
            return numpy.zeros_like(T_gas)
        return self.alpha_RR(T_gas) * k_B * T_gas * radiative_recombination_energy_factor(T_gas, self.charge)

    def alpha_DR(self, T_gas):
        """二電子再結合係数 (このイオン → 1 階下) [cm^3 s^-1]"""
        if self.rr_type is None or len(self.dr_energies) == 0:
            return numpy.zeros_like(numpy.asarray(T_gas, dtype=float))
        return dielectronic_recombination_rate(T_gas, self.dr_energies, self.dr_coefficients)

    def dr_cooling(self, T_gas):
        """二電子再結合による冷却係数 [erg cm^3 s^-1]"""
        if self.rr_type is None or len(self.dr_energies) == 0:
            return numpy.zeros_like(numpy.asarray(T_gas, dtype=float))
        return dielectronic_recombination_cooling(T_gas, self.dr_energies, self.dr_coefficients)

    def q_CI(self, T_gas):
        """衝突電離係数 (このイオン → 1 階上) [cm^3 s^-1]"""
        return direct_ionization_rate(T_gas, self.di_shells)


class ElementData:
    """
    元素の全電離段階のデータ。ions[j] は電荷 j のイオン (j = 0, ..., Z)。
    FAC の光電離データがあるのは電子数 ≤ 10 のイオンだけなので、
    min_charge（光電離データのある最低の電荷）より低い電離段階は計算で扱わない。
    """
    def __init__(self, element):
        self.element = element
        self.Z = ATOMIC_NUMBER[element]
        self.ions = [IonData(element, charge) for charge in range(self.Z + 1)]
        charges_with_data = [ion.charge for ion in self.ions if ion.has_photoionization_data]
        if not charges_with_data:
            raise ValueError(f"No FAC photoionization data for {element}")
        self.min_charge = min(charges_with_data)
        for ion in self.ions[self.min_charge:self.Z]:
            if not ion.has_photoionization_data:
                raise ValueError(f"Photoionization data of {element} are not contiguous in charge ({ion.symbol} is missing)")

    @property
    def number_of_stages(self):
        return self.Z + 1

    def symbols(self):
        return [ion.symbol for ion in self.ions]


_element_cache = {}


def get_element(element):
    """元素のデータを読み込む（キャッシュする）"""
    if element not in _element_cache:
        _element_cache[element] = ElementData(element)
    return _element_cache[element]
