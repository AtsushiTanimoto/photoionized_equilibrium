"""
光電離平衡（電離平衡 + 熱平衡）を計算する PhotoionizationModel クラスの定義

全元素・全電離段階を共通に扱う。各セルで、光子パケットのパスレングス・エスティメータから
局所の光子フラックス（エネルギービンごと）を求め、
  電離平衡: 光電離 + 衝突電離 vs 放射再結合 + 二電子再結合
  熱平衡:   光電離加熱 + Compton 加熱 vs 輝線冷却 + 再結合冷却 (RR, DR) + 衝突電離冷却 + 制動放射 + Compton 冷却
を局所反復で解く。
"""
import astropy.constants
import astropy.units
import numpy
import os
import scipy.optimize
import time

import constants
import atomic_data
import bremsstrahlung
import charge_exchange
import line_cooling
import recombination

# 物理定数 (cgs、astropy.constants の CODATA 値)
k_B = astropy.constants.k_B.cgs.value  # ボルツマン定数 [erg K^-1]
eV = astropy.units.eV.to(astropy.units.erg)  # 電子ボルト [erg]
sigma_T = astropy.constants.sigma_T.cgs.value  # トムソン断面積 [cm^2]
m_e_c2_eV = (astropy.constants.m_e * astropy.constants.c**2).to(astropy.units.eV).value  # 電子の静止エネルギー [eV]
pc = astropy.constants.pc.cgs.value  # パーセク [cm]


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")   # 計算結果の TSV の出力先

# 再結合係数・衝突電離係数の表の温度グリッド [K] (log10 T = 0, ..., 9)
LOG_T_RATE_GRID = numpy.linspace(0.0, 9.0, 451)
T_RATE_GRID = 10**LOG_T_RATE_GRID
T_MIN = 10.0      # 熱収支を解く温度範囲 [K]
T_MAX = 1.0e9
THERMAL_T_GRID = numpy.geomspace(T_MIN, T_MAX, 161)   # 根の探索グリッド (0.05 dex)


def _interpolate_log_table(T, table):
    """log T について線形内挿する（table は形状 (N, len(T_RATE_GRID)) の log10 値）。戻り値は長さ N。"""
    log_T = numpy.clip(numpy.log10(T), LOG_T_RATE_GRID[0], LOG_T_RATE_GRID[-1])
    k = min(max(int(numpy.searchsorted(LOG_T_RATE_GRID, log_T)), 1), len(LOG_T_RATE_GRID) - 1)
    w = (log_T - LOG_T_RATE_GRID[k - 1]) / (LOG_T_RATE_GRID[k] - LOG_T_RATE_GRID[k - 1])
    return 10**((1.0 - w) * table[:, k - 1] + w * table[:, k])


def _log10_table(values):
    return numpy.log10(numpy.maximum(values, 1e-300))


class PhotoionizationModel:
    """光電離平衡を計算するクラス（任意の元素の組）"""
    def __init__(self, radius_edges, hydrogen_number_density, source,
                 elements=constants.BENCHMARK_ELEMENTS, abundances=constants.ABUNDANCES_ANGR,
                 T_gas=1e4, N_nu=2000, E_min_eV=None, E_max_eV=None,
                 recombination_case="A", N_levels_max=None, auger=True, diffuse=False, line_transfer=False, turbulence_km_s=0.0,
                 excitation_autoionization=True):
        """
        radius_edges:            半径セルの境界 [cm]（長さ N_r + 1、昇順）
        hydrogen_number_density: 全水素数密度 [cm^-3]（一様）
        source:                  光源 (Source)
        elements:                扱う元素記号のリスト
        abundances:              H に対する数密度比の辞書
        T_gas:                   温度の初期値 [K]
        N_nu:                    エネルギービンの数
        E_min_eV, E_max_eV:      エネルギービンの範囲 [eV]（省略時は最低の電離閾値から光源の上限まで）
        recombination_case:      "A" (CHIANTI の RR+DR) または "B" (H, He のみ Case B の近似式に置き換える; HII 領域向け)
        N_levels_max:            輝線冷却で考慮する準位数（省略時は N_LEVELS_MAX）
        excitation_autoionization: 衝突電離に励起自動電離 (CHIANTI の easplups) を含めるか
        """
        self.radius_edges = numpy.asarray(radius_edges, dtype=float)
        self.N_r = len(self.radius_edges) - 1
        self.r_cm = 0.5 * (self.radius_edges[1:] + self.radius_edges[:-1])
        self.dr_cm = numpy.diff(self.radius_edges)
        self.r_pc = self.r_cm / pc
        self.hydrogen_number_density = hydrogen_number_density
        self.source = source
        self.element_symbols = list(elements)
        self.abundances = {symbol: abundances[symbol] for symbol in self.element_symbols}
        self.T_gas_init = float(T_gas)
        self.N_nu = N_nu
        self.E_min_eV = E_min_eV
        self.E_max_eV = E_max_eV
        self.recombination_case = recombination_case
        self.N_levels_max = N_levels_max
        self.excitation_autoionization = excitation_autoionization   # 衝突電離に励起自動電離を含めるか
        self.auger = auger              # 内殻光電離に続く Auger 電離 (2 段電離と Auger 電子の加熱) を入れるか
        self.diffuse = diffuse          # H, He の拡散光を外向き近似で輸送するか (True なら recombination_case は "A")
        self.line_transfer = line_transfer   # 輝線の光学的厚さを脱出確率で入れるか (輝線冷却と輝線放射に効く)
        self.turbulence_km_s = turbulence_km_s   # 輝線のドップラー幅に足す乱流速度 [km/s]
        self.line_betas = None          # セルごと・イオンごとの脱出確率 (line_transfer=True のとき 2 回目以降のグローバル反復で更新)
        if diffuse and recombination_case != "A":
            raise ValueError("diffuse=True では拡散光を陽に扱うので recombination_case='A' にする")

        # 原子データ
        self.elements = [atomic_data.get_element(symbol) for symbol in self.element_symbols]
        # 全イオンの一覧（元素ごとに電荷 min_charge, ..., Z）
        self.ion_element_index = []   # イオン → 元素のインデックス
        self.ion_charge = []          # イオン → 電荷
        self.ion_offset = []          # 元素 → ion_list 内の最初のイオン（電荷 min_charge）の位置
        for e, element in enumerate(self.elements):
            self.ion_offset.append(len(self.ion_charge))
            for charge in range(element.min_charge, element.Z + 1):
                self.ion_element_index.append(e)
                self.ion_charge.append(charge)
        self.ion_element_index = numpy.array(self.ion_element_index)
        self.ion_charge = numpy.array(self.ion_charge, dtype=float)
        self.N_ions = len(self.ion_charge)
        self.ions = [self.elements[e].ions[int(j)] for e, j in zip(self.ion_element_index, self.ion_charge)]
        self.ion_symbols = [ion.symbol for ion in self.ions]
        self.ion_ip_erg = numpy.array([ion.ip_eV * eV if ion.ip_eV is not None else 0.0 for ion in self.ions])
        self.element_number_density = numpy.array([hydrogen_number_density * self.abundances[s] for s in self.element_symbols])
        self.ion_element_density = self.element_number_density[self.ion_element_index]

        # 状態: 電離状態の割合 fraction_array[e] は形状 (N_r, Z_e + 1)、列 j が電荷 j
        self.fraction_array = []
        for element in self.elements:
            fraction = numpy.zeros((self.N_r, element.Z + 1))
            fraction[:, element.min_charge] = 1.0   # 初期状態: 扱う最低の電離段階
            self.fraction_array.append(fraction)
        self.temperature_array = numpy.full(self.N_r, self.T_gas_init)
        self.electron_density_array = numpy.zeros(self.N_r)
        self.heating_array = {}
        self.cooling_array = {}
        self.gamma_array = numpy.zeros((self.N_r, self.N_ions))   # 各イオンの光電離率 [s^-1]

        self.E_edges = None
        self.rate_tables_ready = False

    # ------------------------------------------------------------------
    # 準備
    # ------------------------------------------------------------------
    def setup_grids(self):
        """エネルギービン、断面積の行列、再結合・衝突電離係数の表、輝線冷却のデータを準備する"""
        thresholds = [ion.ip_eV for ion in self.ions if ion.has_photoionization_data]
        E_min = self.E_min_eV if self.E_min_eV is not None else 0.999 * min(thresholds)
        E_max = self.E_max_eV if self.E_max_eV is not None else self.source.E_max_eV
        self.E_edges = numpy.geomspace(E_min, E_max, self.N_nu + 1)
        self.E_eV = numpy.sqrt(self.E_edges[1:] * self.E_edges[:-1])   # ビンの代表エネルギー（幾何平均）
        self.E_erg = self.E_eV * eV

        # 光電離断面積 σ [cm^2] と加熱断面積 Σ σ_k (E - I_k) [erg cm^2] の行列 (N_ions, N_nu)
        self.sigma_matrix = numpy.array([ion.photoionization_cross_section(self.E_eV) for ion in self.ions])
        self.heating_matrix = numpy.array([ion.photoionization_heating_cross_section(self.E_eV) for ion in self.ions])
        # Auger 電離を伴う (2 段上に電離する) 分の断面積 [cm^2] (auger=False なら 0、加熱も光電子の分だけにする)
        self.auger_matrix = numpy.array([ion.auger_cross_section(self.E_eV) for ion in self.ions])
        if not self.auger:
            self.auger_matrix[:] = 0.0
            for k, ion in enumerate(self.ions):
                if ion.channels is not None:
                    no_auger = [(I, S0, G, t, 0.0, 0.0) for I, S0, G, t, P_A, E_A in ion.channels]
                    self.heating_matrix[k] = atomic_data.photoionization_heating_cross_section(self.E_eV, no_auger)

        # Compton 散乱: 1 回の散乱で電子が得る平均エネルギー E^2/(m_e c^2) (1 - 21E/(5 m_e c^2)) と失うエネルギー 4kT E/(m_e c^2)
        x = self.E_eV / m_e_c2_eV
        self.compton_heating_weight = sigma_T * self.E_erg * x * numpy.maximum(1.0 - 21.0 / 5.0 * x, 0.0)  # [erg cm^2]
        self.compton_cooling_weight = sigma_T * 4.0 * x                                                      # [cm^2]（× kT）

        # 光源: ビンごとの光子放出率 [s^-1]
        self.Q_bin = self.source.photon_rate_per_bin(self.E_edges)
        self.Q_tot = numpy.sum(self.Q_bin)

        # 再結合係数・衝突電離係数の表（温度グリッド上、log10）
        self._setup_rate_tables()

        # 輝線冷却（CHIANTI のデータがあるイオンのみ）
        names = [ion.chianti_name if ion.charge < ion.Z else "none" for ion in self.ions]
        kwargs = {} if self.N_levels_max is None else {"N_levels_max": self.N_levels_max}
        self.line_cooling = line_cooling.LineCoolingSet(names, **kwargs)
        if self.line_cooling.missing:
            print(f"  line cooling data are missing for: {' '.join(n for n in self.line_cooling.missing if n != 'none')}", flush=True)

    def _setup_rate_tables(self):
        """再結合 (RR, DR)、二電子再結合冷却、衝突電離の係数を温度グリッド上に表にする"""
        T = T_RATE_GRID
        alpha_RR = numpy.array([ion.alpha_RR(T) for ion in self.ions])
        alpha_DR = numpy.array([ion.alpha_DR(T) for ion in self.ions])
        rr_cooling = numpy.array([ion.rr_cooling(T) for ion in self.ions])
        dr_cooling = numpy.array([ion.dr_cooling(T) for ion in self.ions])
        q_CI = numpy.array([ion.q_CI(T, self.excitation_autoionization) for ion in self.ions])   # 直接電離 + 励起自動電離
        # H との電荷交換 (Kingdon & Ferland 1996): k_CX_rec はこのイオン + H I → 1 階下、k_CX_ion はこのイオン + H II → 1 階上
        k_CX_rec = numpy.array([charge_exchange.recombination_rate(ion.Z, ion.charge, T) for ion in self.ions])
        k_CX_ion = numpy.array([charge_exchange.ionization_rate(ion.Z, ion.charge, T) for ion in self.ions])
        if self.recombination_case == "B":
            # HII 領域向け: H II → H I, He II → He I, He III → He II を Case B の近似式に置き換える
            case_B = {"H01": recombination.HI_alphaB, "He02": recombination.HeI_alphaB, "He01": recombination.HeII_alphaB}
            for k, ion in enumerate(self.ions):
                if ion.charge > 0:
                    lower_symbol = self.ions[k - 1].symbol
                    if lower_symbol in case_B:
                        alpha_RR[k] = case_B[lower_symbol](T)
                        rr_cooling[k] = alpha_RR[k] * k_B * T * atomic_data.radiative_recombination_energy_factor(T, ion.charge)
                        alpha_DR[k] = 0.0
                        dr_cooling[k] = 0.0
        self.alpha_RR_table = _log10_table(alpha_RR)
        self.alpha_DR_table = _log10_table(alpha_DR)
        if self.diffuse:
            self._setup_diffuse_tables(T, alpha_RR + alpha_DR)
        self.rr_cooling_table = _log10_table(rr_cooling)
        self.dr_cooling_table = _log10_table(dr_cooling)
        self.q_CI_table = _log10_table(q_CI)
        # 制動放射: イオンごとの Z² ⟨g_ff⟩(Z, T) (Z はイオンの電荷、中性は 0)。Λ_ff = Λ_ff_0 √T n_e Σ n_ion Z² ⟨g_ff⟩
        brems = numpy.array([ion.charge**2 * bremsstrahlung.total_gaunt_factor(ion.charge, T) if ion.charge > 0 else numpy.zeros_like(T)
                             for ion in self.ions])
        self.brems_table = _log10_table(brems)
        self.k_CX_rec_table = _log10_table(k_CX_rec)
        self.k_CX_ion_table = _log10_table(k_CX_ion)
        self.rate_tables_ready = True

    # ------------------------------------------------------------------
    # 拡散光 (外向き近似)
    # ------------------------------------------------------------------
    # 拡散光の成分: (再結合するイオン, 再結合でできるイオン, Case B の α の関数, 再結合線 [(エネルギー [eV], Case B 再結合 1 回あたりの光子数)])
    #   基底状態への再結合連続光 (α_1 = α_A − α_B、Milne の関係の形) と、Case B の再結合のあとに出る H を電離できる線:
    #   He I: 3/4 が三重項で 2³S → 1¹S の 19.82 eV、1/4 が一重項でそのうち 2/3 が 2¹P → 1¹S の 21.22 eV (残りは 2 光子、扱わない)。
    #   He II: 2/3 が Lyα 40.81 eV (残り 1/3 は 2s からの 2 光子、扱わない)。
    #   He II はさらに、Case B の再結合の 1/3 が 2s → 1s の 2 光子放射 (0〜40.81 eV) になり、n = 2 への直接の再結合
    #   (α_2、Balmer 連続光、13.61 eV 以上で H を電離できる) を加える。H I の 2 光子と Balmer 連続光は H を電離できないので扱わない。
    DIFFUSE_COMPONENTS = [("H00", "H01", recombination.HI_alphaB, [], None, None),
                          ("He01", "He02", recombination.HeI_alphaB, [(19.82, 0.75), (21.22, 0.25 * 2.0 / 3.0)], None, None),
                          ("He00", "He01", recombination.HeII_alphaB, [(40.81, 2.0 / 3.0)], (40.81, 1.0 / 3.0), (13.606, recombination.HeII_alpha2))]

    def _setup_diffuse_tables(self, T, alpha_A):
        """拡散光の成分ごとに、基底状態への再結合係数 α_1 = α_A − α_B と Case B の α_B を温度表にする"""
        self.diffuse_components = []
        for upper_symbol, lower_symbol, alpha_B_function, lines, two_photon, balmer in self.DIFFUSE_COMPONENTS:
            if upper_symbol not in self.ion_symbols or lower_symbol not in self.ion_symbols:
                continue
            k_upper = self.ion_symbols.index(upper_symbol)
            k_lower = self.ion_symbols.index(lower_symbol)
            alpha_B = alpha_B_function(T)
            alpha_1 = numpy.maximum(alpha_A[k_upper] - alpha_B, 0.0)
            line_bins = [(int(numpy.searchsorted(self.E_edges, E) - 1), n) for E, n in lines if self.E_edges[0] < E < self.E_edges[-1]]
            component = {"upper": k_upper, "lower": k_lower, "ip_eV": self.ions[k_lower].ip_eV,
                         "alpha_1": _log10_table(alpha_1[None, :]), "alpha_B": _log10_table(alpha_B[None, :]), "lines": line_bins,
                         "two_photon": None, "balmer": None}
            if two_photon is not None:
                # ビンごとの光子数の分布 (2 光子の崩壊 1 回あたり、ビン幅で積分) × Case B の再結合 1 回あたりの崩壊数
                E_max, decays_per_recombination = two_photon
                y_edges = numpy.clip(self.E_edges / E_max, 0.0, 1.0)
                y_fine = numpy.linspace(0.0, 1.0, 4001)
                cumulative = numpy.concatenate([[0.0], numpy.cumsum(0.5 * (recombination.two_photon_distribution(y_fine[1:])
                                                                           + recombination.two_photon_distribution(y_fine[:-1])) * numpy.diff(y_fine))])
                photons_per_bin = numpy.diff(numpy.interp(y_edges, y_fine, cumulative))
                component["two_photon"] = photons_per_bin * decays_per_recombination
            if balmer is not None:
                E_threshold, alpha_2_function = balmer
                component["balmer"] = (E_threshold, _log10_table(alpha_2_function(T)[None, :]))
            self.diffuse_components.append(component)

    def _diffuse_emission(self, T, n_e, n_ion):
        """
        温度 T、電子密度 n_e、イオン密度 n_ion のセルの拡散光の放出率 [photons s^-1 cm^-3] (エネルギービンごと)。
        再結合連続光の形は Milne の関係 j(E) ∝ σ(E) E² exp(−(E − I)/kT) (E ≥ I) をビン幅で積分し、合計が n_e n_ion α_1 になるよう規格化する。
        """
        emission = numpy.zeros(self.N_nu)
        kT_eV = k_B * T / eV
        dE = numpy.diff(self.E_edges)
        for component in self.diffuse_components:
            n_upper = n_ion[component["upper"]]
            if n_upper <= 0.0:
                continue
            rate_1 = n_e * n_upper * _interpolate_log_table(T, component["alpha_1"])[0]
            if rate_1 > 0.0:
                shape = self.sigma_matrix[component["lower"]] * self.E_eV**2 \
                    * numpy.exp(-numpy.minimum((self.E_eV - component["ip_eV"]) / kT_eV, 700.0)) * dE
                shape[self.E_eV < component["ip_eV"]] = 0.0
                total = numpy.sum(shape)
                if total > 0.0:
                    emission += rate_1 * shape / total
            rate_B = n_e * n_upper * _interpolate_log_table(T, component["alpha_B"])[0]
            for k, photons_per_recombination in component["lines"]:
                emission[k] += rate_B * photons_per_recombination
            if component["two_photon"] is not None:
                emission += rate_B * component["two_photon"]
            if component["balmer"] is not None:
                # n = 2 への再結合連続光: 水素様の σ ∝ E⁻³ の Milne の形 j(E) ∝ E⁻¹ exp(−(E − E_2)/kT) (E ≥ E_2)
                E_2, alpha_2_table = component["balmer"]
                rate_2 = n_e * n_upper * _interpolate_log_table(T, alpha_2_table)[0]
                shape = numpy.where(self.E_eV >= E_2, numpy.exp(-numpy.minimum((self.E_eV - E_2) / kT_eV, 700.0)) / self.E_eV, 0.0) * dE
                if numpy.sum(shape) > 0.0:
                    emission += rate_2 * shape / numpy.sum(shape)
        return emission

    def rates_at(self, T):
        """温度 T での全イオンの (α_RR, α_DR, Λ_RR/(n_e n), Λ_DR/(n_e n), q_CI)"""
        return (_interpolate_log_table(T, self.alpha_RR_table),
                _interpolate_log_table(T, self.alpha_DR_table),
                _interpolate_log_table(T, self.rr_cooling_table),
                _interpolate_log_table(T, self.dr_cooling_table),
                _interpolate_log_table(T, self.q_CI_table))

    def charge_exchange_rates_at(self, T):
        """温度 T での全イオンの H との電荷交換係数 (k_CX_rec: + H I → 1 階下, k_CX_ion: + H II → 1 階上) [cm³ s⁻¹]"""
        return (_interpolate_log_table(T, self.k_CX_rec_table),
                _interpolate_log_table(T, self.k_CX_ion_table))

    # ------------------------------------------------------------------
    # 電離平衡
    # ------------------------------------------------------------------
    def _ion_fraction_vector(self, i):
        """セル i の全イオンの割合（ion_list の順）"""
        x = numpy.empty(self.N_ions)
        for e, element in enumerate(self.elements):
            k = self.ion_offset[e]
            n = element.Z + 1 - element.min_charge
            x[k:k + n] = self.fraction_array[e][i, element.min_charge:]
        return x

    def _set_ion_fraction_vector(self, i, x):
        for e, element in enumerate(self.elements):
            k = self.ion_offset[e]
            n = element.Z + 1 - element.min_charge
            self.fraction_array[e][i, :] = 0.0
            self.fraction_array[e][i, element.min_charge:] = x[k:k + n]

    def _solve_ionization_balance(self, up_rates, down_rates, up2_rates=None):
        """
        電離平衡から各電離状態の割合を計算する。
          up_rates[j]:   電荷 j → j+1 の電離率 [s^-1] (光電離のうち 1 段だけの分 + 衝突電離 + 電荷交換)
          down_rates[j]: 電荷 j+1 → j の再結合率 [s^-1] (n_e α + 電荷交換)
          up2_rates[j]:  電荷 j → j+2 の電離率 [s^-1] (内殻光電離に続く Auger 電離。None なら 0)
        戻り値: [x_0, ..., x_N] (N = len(up_rates))
        再結合は 1 段ずつなので、電荷 j と j+1 の境界をまたぐ流束のつり合い
          x_{j+1} down_j = x_j (up_j + up2_j) + x_{j-1} up2_{j-1}
        から下から順に厳密に求まる (行列を解かない)。オーバーフローを避けるため対数で扱う。
        電離率がゼロなら低電離側に、再結合率がゼロなら高電離側に偏る。
        """
        N = len(up_rates)
        if up2_rates is None:
            up2_rates = numpy.zeros(N)
        log_x = numpy.full(N + 1, -numpy.inf)
        log_x[0] = 0.0
        with numpy.errstate(divide="ignore"):
            log_up = numpy.log(numpy.asarray(up_rates, dtype=float) + numpy.asarray(up2_rates, dtype=float))
            log_up2 = numpy.log(numpy.asarray(up2_rates, dtype=float))
            log_down = numpy.log(numpy.asarray(down_rates, dtype=float))
        for j in range(N):
            flux = log_x[j] + log_up[j]
            if j > 0:
                flux = numpy.logaddexp(flux, log_x[j - 1] + log_up2[j - 1])
            if not numpy.isfinite(flux):
                break                                   # これより上には電離しない
            if not numpy.isfinite(log_down[j]):
                log_x[:j + 1] = -numpy.inf              # 再結合しない: これより下は空になる
                log_x[j + 1] = 0.0
                continue
            log_x[j + 1] = flux - log_down[j]
        log_x -= numpy.max(log_x)
        x = numpy.exp(log_x)
        return x / numpy.sum(x)

    def _hydrogen_densities(self, n_ion):
        """全イオンの数密度ベクトルから (n_HI, n_HII) を返す (H がなければ 0)"""
        n_HI = n_ion[self.ion_symbols.index("H01")] if "H01" in self.ion_symbols else 0.0
        n_HII = n_ion[self.ion_symbols.index("H00")] if "H00" in self.ion_symbols else 0.0
        return n_HI, n_HII

    def _solve_all_elements(self, Gamma, n_e, T, n_HI=0.0, n_HII=0.0, Gamma_auger=None):
        """
        全元素の電離平衡を解いて、全イオンの割合（ion_list の順）を返す。
        上向き 1 段: 光電離 (Γ − Γ_Auger) + 衝突電離 n_e q_CI + H II との電荷交換 n_HII k_CX_ion、
        上向き 2 段: 内殻光電離に続く Auger 電離 Γ_Auger (Gamma_auger が None なら 0)、
        下向き: 再結合 n_e (α_RR + α_DR) + H I との電荷交換 n_HI k_CX_rec。
        電荷交換の H の密度は呼び出し側が直前の電離状態から渡す (局所反復で自己無撞着になる)。
        """
        alpha_RR, alpha_DR, _, _, q_CI = self.rates_at(T)
        k_CX_rec, k_CX_ion = self.charge_exchange_rates_at(T)
        alpha = alpha_RR + alpha_DR
        if Gamma_auger is None:
            Gamma_auger = numpy.zeros_like(Gamma)
        x = numpy.empty(self.N_ions)
        for e, element in enumerate(self.elements):
            k = self.ion_offset[e]
            n = element.Z + 1 - element.min_charge
            up2 = Gamma_auger[k:k + n - 1].copy()
            up2[-1] = 0.0                                # 裸の 1 つ手前 (水素様) から 2 段は電離できない
            up = Gamma[k:k + n - 1] - up2 + n_e * q_CI[k:k + n - 1] + n_HII * k_CX_ion[k:k + n - 1]
            down = n_e * alpha[k + 1:k + n] + n_HI * k_CX_rec[k + 1:k + n]
            x[k:k + n] = self._solve_ionization_balance(numpy.maximum(up, 0.0), down, up2)
        return x

    # ------------------------------------------------------------------
    # 本体
    # ------------------------------------------------------------------
    def calculate_equilibrium(self, max_global_iterations=5, N_packets=200000, max_local_iterations=100,
                              relaxation=0.5, seed=0, fraction_tolerance=1e-4, temperature_tolerance=1e-3,
                              fixed_temperature=None, min_global_iterations=2,
                              global_fraction_tolerance=1e-3, global_temperature_tolerance=1e-3, average_estimators=True):
        """
        モンテカルロ法（連続吸収法）で光電離平衡を計算する。
        反復の終了条件:
          グローバル反復 (パケットを内側から外側へ 1 回伝搬させる単位): 前回からの変化の全セル平均
            (各セルでイオン割合の変化の最大値 (絶対値) と温度の相対変化をとり、セルについて平均する) が
            イオン割合 < global_fraction_tolerance かつ 温度 < global_temperature_tolerance になったら終える
            (ただし最低 min_global_iterations 回は回す。1 回目は初期値からの変化なので判定しない)。
            満たさなければ max_global_iterations 回で終える。電離境界など一部のセルの揺らぎで判定が決まらないよう平均をとる。
          モンテカルロの統計誤差を減らすため、各セルに入射するパケットの重み (ビンごと) を反復を通して平均し、
          その平均から局所の光子フラックスを求める (average_estimators=True。反復 n 回で実効パケット数が n 倍になる)。
          セル内の局所反復: イオン割合の変化 < fraction_tolerance かつ 温度の相対変化 < temperature_tolerance で終える
            (max_local_iterations 回で打ち切り)。
        fixed_temperature を与えると熱収支は解かず、その温度で電離平衡だけを解く
        （加熱・冷却の内訳は保存されるので、XSTAR などの温度での収支の診断に使える）。

        光子パケットは光源のスペクトルからサンプリングし、エネルギービンごとに数を数える。
        各セルで全イオンの吸収で重みが exp(-τ) 減衰する。各セルの局所の光子フラックス（ビンごと）
        c_E = Q_p W_E φ(τ) dr / V から光電離率と加熱率を求め、熱収支 → 電離平衡 の順に局所反復する。
        散乱（Compton 散乱による光子の輸送）は考慮しない。
        """
        if self.E_edges is None:
            self.setup_grids()
        rng = numpy.random.default_rng(seed)
        print(f"Starting calculation ({N_packets} packets/iteration, elements: {' '.join(self.element_symbols)}, "
              f"{self.N_ions} ions, {self.N_nu} energy bins)...", flush=True)
        time_start_total = time.time()
        V = 4.0 * numpy.pi * self.r_cm**2 * self.dr_cm
        Q_p = self.Q_tot / N_packets
        print(f"  N_r = {self.N_r} cells (r = {self.radius_edges[0]:.4e}-{self.radius_edges[-1]:.4e} cm), "
              f"n_H = {self.hydrogen_number_density:.3e} cm^-3, "
              f"Q({self.E_edges[0]:.2f}-{self.E_edges[-1]:.3e} eV) = {self.Q_tot:.3e} s^-1", flush=True)
        progress_interval = max(self.N_r // 20, 1)

        self.global_converged = False
        self.line_betas = None
        self.global_iterations_done = 0
        for global_it in range(max_global_iterations):
            if self.line_transfer and global_it > 0:
                self._update_line_betas()   # 前回の反復の状態から輝線の光学的厚さと脱出確率を求める
            previous_fractions = [fraction.copy() for fraction in self.fraction_array]
            previous_T = self.temperature_array.copy()
            W = self.source.sample_packet_counts(self.Q_bin, N_packets, rng)   # ビンごとのパケット数（重み 1）
            if global_it == 0:
                W_in_sum = numpy.zeros((self.N_r, self.N_nu))   # 各セルに入射したパケットの重みの和 (反復について)
            N_unconverged_cells = 0
            time_start_iter = time.time()
            print(f"Iteration {global_it + 1:2d}/{max_global_iterations}: propagating packets...", flush=True)

            for i in range(self.N_r):
                x_loc = self._ion_fraction_vector(i)
                T_loc = self.temperature_array[i]
                dr = self.dr_cm[i]
                W_in_sum[i] += W
                W_in = W_in_sum[i] / (global_it + 1) if average_estimators else W   # 入射するパケットの重み (反復平均)
                base = Q_p * W_in * dr / V[i]    # [cm^-2 s^-1] (φ をかける前)
                local_converged = False
                last_local_it = 0

                # 初期温度 T_loc と整合する電離状態から始める（初期の電離状態が温度と食い違っていると、
                # 熱収支の根の選択が初期の電離状態に引きずられるため）
                for _ in range(3):
                    n_ion = self.ion_element_density * x_loc
                    n_e = numpy.sum(self.ion_charge * n_ion)
                    tau = (n_ion @ self.sigma_matrix) * dr
                    phi = numpy.where(tau > 1e-8, -numpy.expm1(-numpy.minimum(tau, 700.0)) / numpy.maximum(tau, 1e-300), 1.0 - 0.5 * tau)
                    Gamma = self.sigma_matrix @ (base * phi)
                    Gamma_auger = self.auger_matrix @ (base * phi)
                    if numpy.sum(Gamma) <= 0.0:
                        break
                    x_loc = self._solve_all_elements(Gamma, n_e, T_loc, *self._hydrogen_densities(n_ion), Gamma_auger=Gamma_auger)

                for local_it in range(max_local_iterations):
                    last_local_it = local_it
                    n_ion = self.ion_element_density * x_loc
                    n_e = numpy.sum(self.ion_charge * n_ion)
                    n_HI = n_ion[self.ion_symbols.index("H01")] if "H01" in self.ion_symbols else 0.0

                    # 全光学的厚み τ = κ dr と φ(τ) = (1 - e^{-τ})/τ
                    tau = (n_ion @ self.sigma_matrix) * dr
                    phi = numpy.where(tau > 1e-8, -numpy.expm1(-numpy.minimum(tau, 700.0)) / numpy.maximum(tau, 1e-300), 1.0 - 0.5 * tau)
                    c_E = base * phi                              # 局所の光子フラックス（ビンごと）[cm^-2 s^-1]
                    if self.diffuse:
                        # このセル自身の拡散光: 一様に放出され外向きに平均 dr/2 進む (パスレングス推定 emission × dr/2 × φ(τ/2))
                        diffuse_emission = self._diffuse_emission(T_loc, n_e, n_ion)
                        tau_half = 0.5 * tau
                        phi_half = numpy.where(tau_half > 1e-8, -numpy.expm1(-numpy.minimum(tau_half, 700.0)) / numpy.maximum(tau_half, 1e-300), 1.0 - 0.5 * tau_half)
                        c_E = c_E + diffuse_emission * 0.5 * dr * phi_half

                    # 光電離率 [s^-1] と加熱率 [erg s^-1 cm^-3]
                    Gamma = self.sigma_matrix @ c_E
                    Gamma_auger = self.auger_matrix @ c_E
                    heating_PI_ion = n_ion * (self.heating_matrix @ c_E)
                    heating_PI = numpy.sum(heating_PI_ion)
                    heating_compton = n_e * numpy.sum(self.compton_heating_weight * c_E)
                    cooling_compton_per_kT = n_e * numpy.sum(self.compton_cooling_weight * c_E)   # × kT
                    heating_total = heating_PI + heating_compton

                    if heating_total <= 0.0:
                        # 光子が届かないセル: 最低の電離段階・最低温度
                        x_new = numpy.zeros(self.N_ions)
                        for e, element in enumerate(self.elements):
                            x_new[self.ion_offset[e]] = 1.0
                        T_new = T_MIN
                    else:
                        # --- 熱収支から温度を求める ---
                        line_table = self.line_cooling.table(n_e, n_HI, x_loc, self._cell_betas(i))
                        def cooling_terms(T):
                            alpha_RR, alpha_DR, rr_cooling, dr_cooling, q_CI = self.rates_at(T)
                            kT = k_B * T
                            terms = {
                                "line": numpy.sum(n_ion * self.line_cooling.rate(T, line_table)),
                                "recombination_RR": numpy.sum(n_e * n_ion * rr_cooling),
                                "recombination_DR": numpy.sum(n_e * n_ion * dr_cooling),
                                "collisional_ionization": numpy.sum(n_e * n_ion * q_CI * self.ion_ip_erg),
                                "bremsstrahlung": constants.Lambda_ff_0 * numpy.sqrt(T) * n_e * numpy.sum(n_ion * _interpolate_log_table(T, self.brems_table)),
                                "compton": cooling_compton_per_kT * kT,
                            }
                            return terms

                        def thermal_balance_residual(T):
                            return sum(cooling_terms(T).values()) - heating_total

                        if fixed_temperature is not None:
                            T_new = float(fixed_temperature)
                        else:
                            T_new = self._solve_thermal_balance(thermal_balance_residual, T_loc)

                        # --- 電離平衡 ---
                        x_new = self._solve_all_elements(Gamma, n_e, T_new, *self._hydrogen_densities(n_ion), Gamma_auger=Gamma_auger)

                    # 収束判定
                    if numpy.max(numpy.abs(x_new - x_loc)) < fraction_tolerance and abs(T_new - T_loc) < temperature_tolerance * T_loc:
                        x_loc = x_new
                        T_loc = T_new
                        local_converged = True
                        break

                    # アンダーリラクゼーション（振動を防ぐ）
                    x_loc = (1.0 - relaxation) * x_loc + relaxation * x_new
                    T_loc = numpy.exp((1.0 - relaxation) * numpy.log(T_loc) + relaxation * numpy.log(T_new))

                if not local_converged:
                    N_unconverged_cells += 1

                # 収束した状態を保存
                self._set_ion_fraction_vector(i, x_loc)
                self.temperature_array[i] = T_loc
                n_ion = self.ion_element_density * x_loc
                n_e = numpy.sum(self.ion_charge * n_ion)
                self.electron_density_array[i] = n_e
                self.gamma_array[i] = Gamma
                self._store_heating_cooling(i, T_loc, n_ion, n_e, heating_PI_ion, heating_compton, cooling_terms if heating_total > 0.0 else None)

                # パケットの重みを全光学的厚みで減衰させる
                tau_final = (n_ion @ self.sigma_matrix) * dr
                W = W * numpy.exp(-numpy.minimum(tau_final, 700.0))
                if self.diffuse:
                    # このセルの拡散光を外向きのビームに加える (セル内で平均 τ/2 だけ減衰したもの。重みの単位は Q_p)
                    emission = self._diffuse_emission(T_loc, n_e, n_ion)
                    W = W + emission * V[i] / Q_p * numpy.exp(-numpy.minimum(0.5 * tau_final, 700.0))

                if i % progress_interval == 0 or i == self.N_r - 1:
                    elapsed = time.time() - time_start_iter
                    eta = elapsed / (i + 1) * (self.N_r - i - 1)
                    print(f"    cell {i + 1:5d}/{self.N_r} r = {self.r_cm[i]:.4e} cm | T = {T_loc:.3e} K, n_e = {n_e:.3e} cm^-3, "
                          f"{self._dominant_ions_string(i)} | local its = {last_local_it + 1:3d}, "
                          f"remaining packets = {numpy.sum(W) / N_packets:.3f} | elapsed {elapsed / 60.0:.2f} min, ETA {eta / 60.0:.2f} min", flush=True)

            diff_cell = numpy.max([numpy.max(numpy.abs(f - p), axis=1) for f, p in zip(self.fraction_array, previous_fractions)], axis=0)
            diff_T_cell = numpy.abs(self.temperature_array - previous_T) / previous_T
            max_diff, max_diff_T = numpy.max(diff_cell), numpy.max(diff_T_cell)
            mean_diff, mean_diff_T = numpy.mean(diff_cell), numpy.mean(diff_T_cell)
            print(f"Iteration {global_it + 1:2d}/{max_global_iterations}: change in fractions mean {mean_diff:.2e} (max {max_diff:.2e}), "
                  f"in T mean {mean_diff_T:.2e} (max {max_diff_T:.2e}), cells hitting local iteration limit ({max_local_iterations}) = {N_unconverged_cells}  "
                  f"(elapsed time: {(time.time() - time_start_total) / 60.0:.3e} min)", flush=True)
            self.global_iterations_done = global_it + 1
            self.global_converged = (global_it + 1 >= min_global_iterations and mean_diff < global_fraction_tolerance
                                     and mean_diff_T < global_temperature_tolerance)
            if self.global_converged:
                print(f"Converged after {global_it + 1} global iterations (mean change: fractions < {global_fraction_tolerance:.1e}, "
                      f"T < {global_temperature_tolerance:.1e})", flush=True)
                break
        if not self.global_converged:
            print(f"Not converged within {max_global_iterations} global iterations "
                  f"(mean change: fractions < {global_fraction_tolerance:.1e}, T < {global_temperature_tolerance:.1e} not reached)", flush=True)

    def _cell_betas(self, i):
        """セル i のイオンごとの脱出確率のリスト (line_transfer が無効か 1 回目の反復なら None)"""
        return None if self.line_betas is None else self.line_betas[i]

    def _update_line_betas(self):
        """
        現在の (前回のグローバル反復の) 温度・電離状態から、全輝線の線中心の光学的厚さと脱出確率を求める。
          セル i の光学的厚さ  Δτ_i = 1.497×10⁻¹⁵ f_lu λ[Å] N_l / b[km/s] × (1 − g_l n_u / (g_u n_l))、N_l = n_l Δr_i
          b = √(2kT/m + v_turb²)。内向き τ_in = Σ_{j<i} Δτ_j + Δτ_i/2、外向き τ_out = τ_total − τ_in。
          β = [β(τ_in) + β(τ_out)] / 2 (line_cooling.escape_probability、静的な媒質のドップラー線)。
        準位占有数は前回の β で解き直す。輝線の速度場 (膨張) は考えない。
        """
        amu = astropy.constants.u.cgs.value
        n_HI_index = self.ion_symbols.index("H01") if "H01" in self.ion_symbols else None
        dtau = [[None] * self.N_ions for _ in range(self.N_r)]
        for i in range(self.N_r):
            n_ion = self.ion_element_density * self._ion_fraction_vector(i)
            n_e = self.electron_density_array[i]
            T = self.temperature_array[i]
            n_HI = n_ion[n_HI_index] if n_HI_index is not None else 0.0
            previous = self._cell_betas(i)
            for k, model in enumerate(self.line_cooling.ions):
                if model is None or n_ion[k] <= 0.0:
                    continue
                populations = model.populations_at(n_e, n_HI, T, None if previous is None else previous[k])
                n_l = n_ion[k] * populations[model.line_lower]
                n_u = n_ion[k] * populations[model.line_upper]
                stimulated = numpy.clip(1.0 - (model.g[model.line_lower] * n_u) / (model.g[model.line_upper] * numpy.maximum(n_l, 1e-300)), 0.0, 1.0)
                mass = constants.ATOMIC_MASS[self.ions[k].element] * amu
                b_km_s = numpy.sqrt(2.0 * k_B * T / mass / 1.0e10 + self.turbulence_km_s**2)
                dtau[i][k] = 1.497e-15 * model.line_f * model.line_wavelength_angstrom * n_l * self.dr_cm[i] / b_km_s * stimulated
        betas = [[None] * self.N_ions for _ in range(self.N_r)]
        for k, model in enumerate(self.line_cooling.ions):
            if model is None:
                continue
            zero = numpy.zeros(len(model.line_A))
            column = numpy.array([dtau[i][k] if dtau[i][k] is not None else zero for i in range(self.N_r)])   # (N_r, N_lines)
            cumulative = numpy.cumsum(column, axis=0)
            tau_in = cumulative - 0.5 * column
            tau_out = cumulative[-1] - tau_in
            beta = 0.5 * (line_cooling.escape_probability(tau_in) + line_cooling.escape_probability(tau_out))
            for i in range(self.N_r):
                betas[i][k] = beta[i]
        self.line_betas = betas

    def _solve_thermal_balance(self, residual, T_current):
        """
        熱収支 residual(T) = 冷却 - 加熱 = 0 の根を求める。
        光電離プラズマでは高温解と低温解が共存する（熱的不安定性）ことがあるので、
        温度グリッド THERMAL_T_GRID 上で符号変化をすべて探し、現在の温度 T_current に（log T で）最も近い根を brentq で求める。
        根がなければ、残差の符号に応じて T_MIN または T_MAX を返す。
        """
        values = numpy.array([residual(T) for T in THERMAL_T_GRID])
        sign_change = numpy.nonzero(values[:-1] * values[1:] <= 0.0)[0]
        if len(sign_change) == 0:
            return T_MIN if values[0] > 0.0 else T_MAX
        roots = []
        for k in sign_change:
            if values[k] == 0.0:
                roots.append(THERMAL_T_GRID[k])
                continue
            roots.append(scipy.optimize.brentq(residual, THERMAL_T_GRID[k], THERMAL_T_GRID[k + 1], xtol=1e-3, rtol=1e-6))
        roots = numpy.array(roots)
        return roots[numpy.argmin(numpy.abs(numpy.log(roots) - numpy.log(T_current)))]

    def _dominant_ions_string(self, i, elements=("H", "O", "Fe")):
        parts = []
        for e, element in enumerate(self.elements):
            if element.element in elements:
                j = int(numpy.argmax(self.fraction_array[e][i]))
                parts.append(f"{element.element}:{element.ions[j].symbol}")
        return " ".join(parts)

    def _store_heating_cooling(self, i, T, n_ion, n_e, heating_PI_ion, heating_compton, cooling_terms):
        """セル i の加熱・冷却の内訳 [erg s^-1 cm^-3] を元素ごとに保存する"""
        if not self.heating_array:
            for symbol in self.element_symbols:
                self.heating_array[symbol] = numpy.zeros(self.N_r)
                self.cooling_array[symbol] = numpy.zeros(self.N_r)
            self.heating_array["compton"] = numpy.zeros(self.N_r)
            for key in ("line", "recombination_RR", "recombination_DR", "collisional_ionization", "bremsstrahlung", "compton"):
                self.cooling_array[key] = numpy.zeros(self.N_r)
        for e, symbol in enumerate(self.element_symbols):
            mask = self.ion_element_index == e
            self.heating_array[symbol][i] = numpy.sum(heating_PI_ion[mask])
        self.heating_array["compton"][i] = heating_compton
        if cooling_terms is None:
            return
        terms = cooling_terms(T)
        for key, value in terms.items():
            self.cooling_array[key][i] = value
        # 元素ごとの冷却（輝線 + 再結合 + 衝突電離）
        alpha_RR, alpha_DR, rr_cooling, dr_cooling, q_CI = self.rates_at(T)
        n_HI = n_ion[self.ion_symbols.index("H01")] if "H01" in self.ion_symbols else 0.0
        line_table = self.line_cooling.table(n_e, n_HI, self._ion_fraction_vector(i), self._cell_betas(i))
        per_ion = (n_ion * self.line_cooling.rate(T, line_table) + n_e * n_ion * rr_cooling +
                   n_e * n_ion * dr_cooling + n_e * n_ion * q_CI * self.ion_ip_erg)
        for e, symbol in enumerate(self.element_symbols):
            self.cooling_array[symbol][i] = numpy.sum(per_ion[self.ion_element_index == e])

    # ------------------------------------------------------------------
    # 出力
    # ------------------------------------------------------------------
    def save_tsv(self, filename="simulation.tsv"):
        """
        計算結果を TSV ファイルとして保存する。
        列: Radius_cm, Temperature, n_e, 各イオンの割合（記号は <元素><電子数 2 桁>、裸のイオンは <元素>00）
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUT_DIR, filename)
        columns = [self.r_cm, self.temperature_array, self.electron_density_array]
        header = ["Radius_cm", "Temperature", "n_e"]
        for e, element in enumerate(self.elements):
            for j in range(element.Z + 1):
                columns.append(self.fraction_array[e][:, j])
                header.append(element.ions[j].symbol)
        numpy.savetxt(filepath, numpy.column_stack(columns), delimiter="\t", header="\t".join(header), comments="")
        return filepath

    def save_line_emission_tsv(self, filename="line_emission.tsv", minimum_fraction=1e-8, minimum_photons_per_ionization=1e-6):
        """
        衝突励起による輝線の放射率をセルごと・イオンごと・線ごとに TSV ファイルとして保存する。
        列: cell, ion (<元素><電子数 2 桁>), chianti (CHIANTI のイオン名), upper, lower (CHIANTI の準位番号、1 始まり),
            energy_eV, photons_per_s_per_cm3 (体積放射率), photons_per_photoionization
        photons_per_photoionization = 放射率 / (Γ n_ion) は、MONACO が光電離イベントごとに輝線を抽選するときの
        規格化（再結合 1 回あたりの光子数、.ln の衝突励起成分に相当）。平衡では再結合率 ≈ 光電離率 + 衝突電離率なので、
        衝突電離が効く高温では「再結合 1 回あたり」より大きめになる。
        占有率は各セルの T, n_e, n_HI で解き直す。割合 < minimum_fraction のイオンと、
        光電離 1 回あたり < minimum_photons_per_ionization の線は書かない。
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w") as f:
            f.write("cell\tion\tchianti\tupper\tlower\tenergy_eV\tphotons_per_s_per_cm3\tphotons_per_photoionization\n")
            for i in range(self.N_r):
                x = self._ion_fraction_vector(i)
                n_ion = self.ion_element_density * x
                n_e = self.electron_density_array[i]
                T = self.temperature_array[i]
                n_HI = n_ion[self.ion_symbols.index("H01")] if "H01" in self.ion_symbols else 0.0
                for k, (ion, model) in enumerate(zip(self.ions, self.line_cooling.ions)):
                    if model is None or x[k] < minimum_fraction or n_ion[k] <= 0.0:
                        continue
                    ionization_rate = self.gamma_array[i, k] * n_ion[k]   # [s^-1 cm^-3]
                    if ionization_rate <= 0.0:
                        continue
                    betas = self._cell_betas(i)
                    upper, lower, energy_eV, photons = model.line_emission(n_e, n_HI, T, None if betas is None else betas[k])
                    emissivity = photons * n_ion[k]
                    per_ionization = emissivity / ionization_rate
                    order = numpy.argsort(energy_eV)
                    for m in order:
                        if per_ionization[m] < minimum_photons_per_ionization:
                            continue
                        f.write(f"{i}\t{ion.symbol}\t{ion.chianti_name}\t{upper[m]}\t{lower[m]}\t{energy_eV[m]:.6e}\t{emissivity[m]:.6e}\t{per_ionization[m]:.6e}\n")
        return filepath

    def save_level_population_tsv(self, prefix="level_populations"):
        """
        輝線冷却の多準位モデルで解いた準位占有数をイオンごとに TSV ファイル output/<prefix>/<イオン記号>.tsv として保存する。
        行がセル、列が準位: Radius_cm, Temperature, n_e, n_ion (イオンの数密度 [cm^-3]), level_01, level_02, ...
        （level_k は CHIANTI の準位番号 k の占有率、イオン内で Σ = 1。準位の配置・項・エネルギーは CHIANTI の .elvlc を参照）。
        占有数は衝突励起・脱励起と自発放射だけの統計平衡（光励起・再結合カスケードは含まない）を
        各セルの T, n_e, n_HI で解き直したもの。輝線冷却のデータがないイオンは書かない。
        戻り値: 出力ディレクトリ
        """
        directory = os.path.join(OUTPUT_DIR, prefix)
        os.makedirs(directory, exist_ok=True)
        n_HI_index = self.ion_symbols.index("H01") if "H01" in self.ion_symbols else None
        for k, (ion, model) in enumerate(zip(self.ions, self.line_cooling.ions)):
            if model is None:
                continue
            with open(os.path.join(directory, f"{ion.symbol}.tsv"), "w") as f:
                f.write("Radius_cm\tTemperature\tn_e\tn_ion\t" + "\t".join(f"level_{m + 1:02d}" for m in range(model.N_levels)) + "\n")
                for i in range(self.N_r):
                    n_ion = self.ion_element_density * self._ion_fraction_vector(i)
                    n_e = self.electron_density_array[i]
                    T = self.temperature_array[i]
                    n_HI = n_ion[n_HI_index] if n_HI_index is not None else 0.0
                    betas = self._cell_betas(i)
                    populations = model.populations_at(n_e, n_HI, T, None if betas is None else betas[k])
                    f.write(f"{self.r_cm[i]:.6e}\t{T:.6e}\t{n_e:.6e}\t{n_ion[k]:.6e}\t" + "\t".join(f"{x:.6e}" for x in populations) + "\n")
        return directory

    def save_heating_cooling_tsv(self, filename="heating_cooling.tsv"):
        """加熱・冷却の内訳 [erg s^-1 cm^-3] を TSV ファイルとして保存する"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUT_DIR, filename)
        columns = [self.r_cm, self.temperature_array]
        header = ["Radius_cm", "Temperature"]
        for key, value in self.heating_array.items():
            columns.append(value)
            header.append(f"heating_{key}")
        columns.append(sum(self.heating_array.values()))
        header.append("heating_total")
        for key, value in self.cooling_array.items():
            columns.append(value)
            header.append(f"cooling_{key}")
        columns.append(sum(self.cooling_array[k] for k in ("line", "recombination_RR", "recombination_DR", "collisional_ionization", "bremsstrahlung", "compton")))
        header.append("cooling_total")
        numpy.savetxt(filepath, numpy.column_stack(columns), delimiter="\t", header="\t".join(header), comments="")
        return filepath
