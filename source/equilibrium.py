"""
光電離平衡（電離平衡 + 熱平衡）を計算する PhotoionizationModel クラスの定義

全元素・全電離段階を共通に扱う。各セルで、光子パケットのパスレングス・エスティメータから
局所の光子フラックス（エネルギービンごと）を求め、
  電離平衡: 光電離 + 衝突電離 vs 放射再結合 + 二電子再結合
  熱平衡:   光電離加熱 + Compton 加熱 vs 輝線冷却 + 再結合冷却 (RR, DR) + 衝突電離冷却 + 制動放射 + Compton 冷却
を局所反復で解く。
"""
import numpy
import os
import scipy.optimize
import time

from constants import k_B, eV, sigma_T, m_e_c2_eV, Lambda_ff_0, ABUNDANCES_ANGR, BENCHMARK_ELEMENTS
from atomic_data import get_element, radiative_recombination_energy_factor
from bremsstrahlung import g_ff
from line_cooling import LineCoolingSet
from recombination import HI_alphaB, HeI_alphaB, HeII_alphaB


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# 再結合係数・衝突電離係数の表の温度グリッド [K] (log10 T = 0, ..., 9)
LOG_T_RATE_GRID = numpy.linspace(0.0, 9.0, 451)
T_RATE_GRID = 10**LOG_T_RATE_GRID
T_MIN = 10.0      # 熱収支を解く温度範囲 [K]
T_MAX = 1.0e9


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
                 elements=BENCHMARK_ELEMENTS, abundances=ABUNDANCES_ANGR,
                 T_gas=1e4, N_nu=2000, E_min_eV=None, E_max_eV=None,
                 recombination_case="A", N_levels_max=None):
        """
        radius_edges:            半径セルの境界 [cm]（長さ N_r + 1、昇順）
        hydrogen_number_density: 全水素数密度 [cm^-3]（一様）
        source:                  光源 (source.Source)
        elements:                扱う元素記号のリスト
        abundances:              H に対する数密度比の辞書
        T_gas:                   温度の初期値 [K]
        N_nu:                    エネルギービンの数
        E_min_eV, E_max_eV:      エネルギービンの範囲 [eV]（省略時は最低の電離閾値から光源の上限まで）
        recombination_case:      "A" (CHIANTI の RR+DR) または "B" (H, He のみ Case B の近似式に置き換える; HII 領域向け)
        N_levels_max:            輝線冷却で考慮する準位数（省略時は line_cooling.N_LEVELS_MAX）
        """
        self.radius_edges = numpy.asarray(radius_edges, dtype=float)
        self.N_r = len(self.radius_edges) - 1
        self.r_cm = 0.5 * (self.radius_edges[1:] + self.radius_edges[:-1])
        self.dr_cm = numpy.diff(self.radius_edges)
        self.r_pc = self.r_cm / 3.0856775814913673e18
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

        # 原子データ
        self.elements = [get_element(symbol) for symbol in self.element_symbols]
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
        self.line_cooling = LineCoolingSet(names, **kwargs)
        if self.line_cooling.missing:
            print(f"  line cooling data are missing for: {' '.join(n for n in self.line_cooling.missing if n != 'none')}", flush=True)

    def _setup_rate_tables(self):
        """再結合 (RR, DR)、二電子再結合冷却、衝突電離の係数を温度グリッド上に表にする"""
        T = T_RATE_GRID
        alpha_RR = numpy.array([ion.alpha_RR(T) for ion in self.ions])
        alpha_DR = numpy.array([ion.alpha_DR(T) for ion in self.ions])
        rr_cooling = numpy.array([ion.rr_cooling(T) for ion in self.ions])
        dr_cooling = numpy.array([ion.dr_cooling(T) for ion in self.ions])
        q_CI = numpy.array([ion.q_CI(T) for ion in self.ions])
        if self.recombination_case == "B":
            # HII 領域向け: H II → H I, He II → He I, He III → He II を Case B の近似式に置き換える
            case_B = {"H01": HI_alphaB, "He02": HeI_alphaB, "He01": HeII_alphaB}
            for k, ion in enumerate(self.ions):
                if ion.charge > 0:
                    lower_symbol = self.ions[k - 1].symbol
                    if lower_symbol in case_B:
                        alpha_RR[k] = case_B[lower_symbol](T)
                        rr_cooling[k] = alpha_RR[k] * k_B * T * radiative_recombination_energy_factor(T, ion.charge)
                        alpha_DR[k] = 0.0
                        dr_cooling[k] = 0.0
        self.alpha_RR_table = _log10_table(alpha_RR)
        self.alpha_DR_table = _log10_table(alpha_DR)
        self.rr_cooling_table = _log10_table(rr_cooling)
        self.dr_cooling_table = _log10_table(dr_cooling)
        self.q_CI_table = _log10_table(q_CI)
        self.rate_tables_ready = True

    def rates_at(self, T):
        """温度 T での全イオンの (α_RR, α_DR, Λ_RR/(n_e n), Λ_DR/(n_e n), q_CI)"""
        return (_interpolate_log_table(T, self.alpha_RR_table),
                _interpolate_log_table(T, self.alpha_DR_table),
                _interpolate_log_table(T, self.rr_cooling_table),
                _interpolate_log_table(T, self.dr_cooling_table),
                _interpolate_log_table(T, self.q_CI_table))

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

    def _solve_ionization_balance(self, up_rates, down_rates):
        """
        隣接する電離状態間のバランスから各電離状態の割合を計算する。
          up_rates[j]:   電荷 j → j+1 の電離率 [s^-1] (光電離 + 衝突電離)
          down_rates[j]: 電荷 j+1 → j の再結合率 [s^-1] (n_e α)
        戻り値: [x_0, ..., x_N] (N = len(up_rates))
        電離率がゼロなら低電離側に、再結合率がゼロなら高電離側に偏る。オーバーフローを避けるため対数で扱う。
        """
        N = len(up_rates)
        log_x = numpy.zeros(N + 1)
        for j in range(N):
            if up_rates[j] <= 0.0:
                log_x[j + 1:] = -numpy.inf
                break
            if down_rates[j] <= 0.0:
                log_x[:j + 1] = -numpy.inf
                log_x[j + 1] = 0.0
                continue
            log_x[j + 1] = log_x[j] + numpy.log(up_rates[j]) - numpy.log(down_rates[j])
        log_x -= numpy.max(log_x)
        x = numpy.exp(log_x)
        return x / numpy.sum(x)

    def _solve_all_elements(self, Gamma, n_e, T):
        """全元素の電離平衡を解いて、全イオンの割合（ion_list の順）を返す"""
        alpha_RR, alpha_DR, _, _, q_CI = self.rates_at(T)
        alpha = alpha_RR + alpha_DR
        x = numpy.empty(self.N_ions)
        for e, element in enumerate(self.elements):
            k = self.ion_offset[e]
            n = element.Z + 1 - element.min_charge
            up = Gamma[k:k + n - 1] + n_e * q_CI[k:k + n - 1]
            down = n_e * alpha[k + 1:k + n]
            x[k:k + n] = self._solve_ionization_balance(up, down)
        return x

    # ------------------------------------------------------------------
    # 本体
    # ------------------------------------------------------------------
    def calculate_equilibrium(self, max_global_iterations=5, N_packets=200000, max_local_iterations=100,
                              relaxation=0.5, seed=0, fraction_tolerance=1e-4, temperature_tolerance=1e-3,
                              fixed_temperature=None):
        """
        モンテカルロ法（連続吸収法）で光電離平衡を計算する。
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

        for global_it in range(max_global_iterations):
            previous_fractions = [fraction.copy() for fraction in self.fraction_array]
            previous_T = self.temperature_array.copy()
            W = self.source.sample_packet_counts(self.Q_bin, N_packets, rng)   # ビンごとのパケット数（重み 1）
            N_unconverged_cells = 0
            time_start_iter = time.time()
            print(f"Iteration {global_it + 1:2d}/{max_global_iterations}: propagating packets...", flush=True)

            for i in range(self.N_r):
                x_loc = self._ion_fraction_vector(i)
                T_loc = self.temperature_array[i]
                dr = self.dr_cm[i]
                base = Q_p * W * dr / V[i]    # [cm^-2 s^-1] (φ をかける前)
                local_converged = False
                last_local_it = 0

                for local_it in range(max_local_iterations):
                    last_local_it = local_it
                    n_ion = self.ion_element_density * x_loc
                    n_e = numpy.sum(self.ion_charge * n_ion)
                    n_HI = n_ion[self.ion_symbols.index("H01")] if "H01" in self.ion_symbols else 0.0

                    # 全光学的厚み τ = κ dr と φ(τ) = (1 - e^{-τ})/τ
                    tau = (n_ion @ self.sigma_matrix) * dr
                    phi = numpy.where(tau > 1e-8, -numpy.expm1(-numpy.minimum(tau, 700.0)) / numpy.maximum(tau, 1e-300), 1.0 - 0.5 * tau)
                    c_E = base * phi                              # 局所の光子フラックス（ビンごと）[cm^-2 s^-1]

                    # 光電離率 [s^-1] と加熱率 [erg s^-1 cm^-3]
                    Gamma = self.sigma_matrix @ c_E
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
                        line_table = self.line_cooling.table(n_e, n_HI, x_loc)
                        charge2 = self.ion_charge**2

                        def cooling_terms(T):
                            alpha_RR, alpha_DR, rr_cooling, dr_cooling, q_CI = self.rates_at(T)
                            kT = k_B * T
                            terms = {
                                "line": numpy.sum(n_ion * self.line_cooling.rate(T, line_table)),
                                "recombination_RR": numpy.sum(n_e * n_ion * rr_cooling),
                                "recombination_DR": numpy.sum(n_e * n_ion * dr_cooling),
                                "collisional_ionization": numpy.sum(n_e * n_ion * q_CI * self.ion_ip_erg),
                                "bremsstrahlung": Lambda_ff_0 * g_ff(T) * numpy.sqrt(T) * n_e * numpy.sum(charge2 * n_ion),
                                "compton": cooling_compton_per_kT * kT,
                            }
                            return terms

                        def thermal_balance_residual(T):
                            return sum(cooling_terms(T).values()) - heating_total

                        if fixed_temperature is not None:
                            T_new = float(fixed_temperature)
                        else:
                            try:
                                T_new = scipy.optimize.brentq(thermal_balance_residual, T_MIN, T_MAX, xtol=1e-3, rtol=1e-6)
                            except ValueError:
                                T_new = T_MIN if thermal_balance_residual(T_MIN) > 0.0 else T_MAX

                        # --- 電離平衡 ---
                        x_new = self._solve_all_elements(Gamma, n_e, T_new)

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

                if i % progress_interval == 0 or i == self.N_r - 1:
                    elapsed = time.time() - time_start_iter
                    eta = elapsed / (i + 1) * (self.N_r - i - 1)
                    print(f"    cell {i + 1:5d}/{self.N_r} r = {self.r_cm[i]:.4e} cm | T = {T_loc:.3e} K, n_e = {n_e:.3e} cm^-3, "
                          f"{self._dominant_ions_string(i)} | local its = {last_local_it + 1:3d}, "
                          f"remaining packets = {numpy.sum(W) / N_packets:.3f} | elapsed {elapsed / 60.0:.2f} min, ETA {eta / 60.0:.2f} min", flush=True)

            max_diff = max(numpy.max(numpy.abs(f - p)) for f, p in zip(self.fraction_array, previous_fractions))
            max_diff_T = numpy.max(numpy.abs(self.temperature_array - previous_T) / previous_T)
            print(f"Iteration {global_it + 1:2d}/{max_global_iterations}: max change in fractions = {max_diff:.2e}, "
                  f"in T = {max_diff_T:.2e}, cells hitting local iteration limit ({max_local_iterations}) = {N_unconverged_cells}  "
                  f"(elapsed time: {(time.time() - time_start_total) / 60.0:.3e} min)", flush=True)

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
        line_table = self.line_cooling.table(n_e, n_HI, self._ion_fraction_vector(i))
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
        os.makedirs(DATA_DIR, exist_ok=True)
        filepath = os.path.join(DATA_DIR, filename)
        columns = [self.r_cm, self.temperature_array, self.electron_density_array]
        header = ["Radius_cm", "Temperature", "n_e"]
        for e, element in enumerate(self.elements):
            for j in range(element.Z + 1):
                columns.append(self.fraction_array[e][:, j])
                header.append(element.ions[j].symbol)
        numpy.savetxt(filepath, numpy.column_stack(columns), delimiter="\t", header="\t".join(header), comments="")
        return filepath

    def save_heating_cooling_tsv(self, filename="heating_cooling.tsv"):
        """加熱・冷却の内訳 [erg s^-1 cm^-3] を TSV ファイルとして保存する"""
        os.makedirs(DATA_DIR, exist_ok=True)
        filepath = os.path.join(DATA_DIR, filename)
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
