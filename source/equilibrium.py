"""
光電離平衡を計算する PhotoionizationModel クラスの定義
"""
import numpy
import os
import scipy.integrate
import scipy.optimize
import time

from constants import h, k_B, HI_nu0, HeII_nu0, HeI_nu0, Lambda_ff_0, Y_He
from physics import (g_ff, HI_alphaB, HeII_alphaB, HeI_alphaB, HI_anu, HeII_anu, HeI_anu, B_nu)


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class PhotoionizationModel:
    """光電離平衡を計算するクラス（水素 + ヘリウム）"""
    def __init__(self, radius_grid_array, hydrogen_number_density, T_gas=1e4, T_BB=4e4, R_star_solar=10.0, N_nu=1001):
        self.radius_grid_array          = radius_grid_array
        self.hydrogen_number_density    = hydrogen_number_density            # 全水素数密度 [cm^-3]
        self.helium_number_density      = hydrogen_number_density * Y_He    # 全ヘリウム数密度 [cm^-3]
        self.T_gas_init                 = T_gas   # ガスの初期温度 [K]
        self.T_BB                       = T_BB          # 星の黒体放射温度 [K]
        self.R_star_cm                  = R_star_solar * 6.96e10  # 星の半径 [cm]
        self.N_nu                       = N_nu          # 振動数グリッドの分割数
        self.nu_grid                    = None
        self.a_nu_grid                  = None
        self.B_nu_grid                  = None
        self.R_s_cm                     = None
        self.R_s_pc                     = None
        self.N_r                        = len(self.radius_grid_array)
        self.r_cm                       = None
        self.r_pc                       = None
        self.HI_fraction_array          = numpy.ones(self.N_r)
        self.HII_fraction_array         = numpy.zeros(self.N_r)
        self.HeI_fraction_array         = numpy.ones(self.N_r)
        self.HeII_fraction_array        = numpy.zeros(self.N_r)
        self.HeIII_fraction_array       = numpy.zeros(self.N_r)
        self.temperature_array          = numpy.full(self.N_r, float(self.T_gas_init))

    def _solve_ionization_balance(self, Gammas, alphas, n_e):
        """
        隣接する電離状態間のバランスから、各電離状態の割合を計算する汎用関数。
         Gammas: [Gamma_{0->1}, Gamma_{1->2}, ...] (例: [Gamma_HeI, Gamma_HeII])
         alphas: [alpha_{1->0}, alpha_{2->1}, ...] (例: [alpha_HeI, alpha_HeII])
         n_e: 電子密度
         戻り値: [x_0, x_1, ..., x_N] (中性, 1階電離, 2階電離...)
        """
        N = len(Gammas)
        r = numpy.zeros(N)
        for i in range(N):
            if Gammas[i] > 1e-30:
                if n_e > 0:
                    r[i] = alphas[i] * n_e / Gammas[i]
                else:
                    r[i] = 0.0  # 電子がないため再結合できず、即座に上位の電離状態に移行する
            else:
                r[i] = 1e30  # 光電離が起きない場合は低電離状態に偏る
                
        # ratio[i] = n_i / n_N の比を計算
        ratio = numpy.ones(N + 1)
        for i in range(N - 1, -1, -1):
            ratio[i] = ratio[i+1] * r[i]
            
        # 規格化して割合を返す
        return ratio / numpy.sum(ratio)



    def setup_grids(self):
        """振動数と距離のグリッド、および初期定数を設定する"""
        # radius_grid_array [cm] から距離グリッドを設定
        self.r_cm = numpy.asarray(self.radius_grid_array, dtype=float)
        self.N_r = len(self.r_cm)
        self.r_pc = self.r_cm / 3.086e18

        # 連続スペクトルの積分を離散化するため、振動数のグリッドを作成
        # 対数スケールで nu_0 から 1000*nu_0 まで分割
        self.nu_grid = numpy.linspace(HI_nu0, 1.000e+03*HI_nu0, self.N_nu)
        self.a_nu_grid = HI_anu(self.nu_grid)
        self.B_nu_grid = B_nu(self.nu_grid, self.T_BB)
        
        # 理論的なストロームグレン半径 R_s を事前計算 (比較用)
        # ヘリウムによる電子密度の増加を考慮: n_e ≈ n_H(1 + Y_He) (完全電離時)
        integrand_Q         = numpy.pi * self.B_nu_grid / (h * self.nu_grid)
        Q_H0                = 4 * numpy.pi * self.R_star_cm**2 * numpy.trapezoid(integrand_Q, self.nu_grid)
        n_e_fully_ionized   = self.hydrogen_number_density * (1.0 + Y_He)
        self.R_s_cm         = (3 * Q_H0 / (4 * numpy.pi * self.hydrogen_number_density * n_e_fully_ionized * HI_alphaB(self.T_gas_init)))**(1/3)
        self.R_s_pc         = self.R_s_cm / 3.086e18
        print(f"理論的なストロームグレン半径 R_s = {self.R_s_pc:.2f} pc")

    def calculate_equilibrium(self, max_global_iterations=30, tolerance=1e-3, N_packets=200000):
        """
        モンテカルロ法を利用して光電離平衡の計算を行う（水素 + ヘリウム）

        光子パケットは H I, He I, He II の3つの吸収体と相互作用する。
        各セルで光学的厚みは全吸収体の和 τ = (n_HI σ_HI + n_HeI σ_HeI + n_HeII σ_HeII) dr で、
        各種への配分は不透明度の寄与比で決まる。
        """        
        print(f"計算を開始します（グローバル反復法: {N_packets}パケット/反復, H+He）...")
        time_start_total = time.time()
        
        integrand_Q = numpy.pi * self.B_nu_grid / (h * self.nu_grid)
        Q_cum = scipy.integrate.cumulative_trapezoid(integrand_Q, self.nu_grid, initial=0)
        Q_tot = 4 * numpy.pi * self.R_star_cm**2 * Q_cum[-1]
        
        if Q_cum[-1] > 0:
            CDF = Q_cum / Q_cum[-1]
        else:
            CDF = numpy.linspace(0, 1, len(self.nu_grid))
            
        dr = self.r_cm[1] - self.r_cm[0] if self.N_r > 1 else 0
        V = numpy.maximum(4 * numpy.pi * self.r_cm**2 * dr, 1e-30)
        Q_p = Q_tot / N_packets
        
        for global_it in range(max_global_iterations):
            previous_HI_fraction = self.HI_fraction_array.copy()
            previous_HeII_fraction = self.HeII_fraction_array.copy()
            
            U_nu = numpy.random.rand(N_packets)
            nu_packets = numpy.interp(U_nu, CDF, self.nu_grid)
            
            # 各パケットの光電離断面積を事前計算
            HI_sigma_all   = HI_anu(nu_packets)
            HeII_sigma_all  = HeII_anu(nu_packets)
            HeI_sigma_all  = HeI_anu(nu_packets)
            
            # 各吸収体による余剰エネルギー（加熱に寄与）
            HI_E_heat_all  = h * numpy.maximum(nu_packets - HI_nu0, 0.0)
            HeII_E_heat_all = h * numpy.maximum(nu_packets - HeII_nu0, 0.0)
            HeI_E_heat_all = h * numpy.maximum(nu_packets - HeI_nu0, 0.0)
            
            # 連続吸収法: パケットは消滅せず重み W が減衰していく
            W = numpy.ones(N_packets)
            active = numpy.ones(N_packets, dtype=bool)
            
            for i in range(self.N_r):
                if not numpy.any(active):
                    self.HII_fraction_array[i]      = 0.0
                    self.HI_fraction_array[i]       = 1.0
                    self.HeIII_fraction_array[i]    = 0.0
                    self.HeII_fraction_array[i]     = 0.0
                    self.HeI_fraction_array[i]      = 1.0
                    self.temperature_array[i]       = 10.0
                    continue
                
                # 局所反復: セル内の電離度・温度と光学的厚みを整合させる
                HII_fraction_loc    = self.HII_fraction_array[i]
                HeIII_fraction_loc   = self.HeIII_fraction_array[i]
                HeII_fraction_loc   = self.HeII_fraction_array[i]
                HeI_fraction_loc   = self.HeI_fraction_array[i]
                T_loc               = self.temperature_array[i]
                HI_sigma           = HI_sigma_all[active]
                HeII_sigma          = HeII_sigma_all[active]
                HeI_sigma          = HeI_sigma_all[active]
                HI_E_heat          = HI_E_heat_all[active]
                HeII_E_heat         = HeII_E_heat_all[active]
                HeI_E_heat         = HeI_E_heat_all[active]
                base_factor         = Q_p * W[active] / V[i]
                
                for _ in range(20):  # 連立方程式なので反復回数を増やす
                    # 数密度の計算
                    HII_number_density      = self.hydrogen_number_density * HII_fraction_loc
                    HI_number_density      = self.hydrogen_number_density * max(1.0 - HII_fraction_loc, 1e-15)
                    HeIII_number_density     = self.helium_number_density * HeIII_fraction_loc
                    HeII_number_density     = self.helium_number_density * HeII_fraction_loc
                    HeI_number_density     = self.helium_number_density * HeI_fraction_loc
                    electron_number_density = HII_number_density + HeII_number_density + 2.0 * HeIII_number_density
                    
                    # 全光学的厚み τ = κ × dr
                    kappa = HI_number_density * HI_sigma + HeI_number_density * HeI_sigma + HeII_number_density * HeII_sigma
                    tau_total = kappa * dr
                    
                    # φ(τ) = (1 - e^{-τ}) / τ: τ→0 で 1 に漸近
                    # 解析的パスレングス・エスティメータの多種拡張
                    phi = numpy.where(
                        tau_total > 1e-8,
                        (1.0 - numpy.exp(-tau_total)) / tau_total,
                        1.0 - tau_total / 2.0)
                    
                    # 各種の光電離率（1原子あたり） [s⁻¹]
                    common = base_factor * phi * dr
                    Gamma_HI = numpy.sum(common * HI_sigma)
                    Gamma_HeI = numpy.sum(common * HeI_sigma)
                    Gamma_HeII = numpy.sum(common * HeII_sigma)
                    
                    # 加熱率（体積あたり） [erg s⁻¹ cm⁻³]
                    # H_vol = Σ_s n_s × Σ_packets (common × σ_s × E_s)
                    H_total_vol = (
                        HI_number_density * numpy.sum(common * HI_sigma * HI_E_heat) +
                        HeI_number_density * numpy.sum(common * HeI_sigma * HeI_E_heat) +
                        HeII_number_density * numpy.sum(common * HeII_sigma * HeII_E_heat))
                    
                    if Gamma_HI + Gamma_HeI + Gamma_HeII <= 0:
                        HII_fraction_new = 0.0
                        HeII_fraction_new = 0.0
                        HeIII_fraction_new = 0.0
                        T_new = 10.0
                    else:
                        # --- 温度を熱収支方程式から求解 ---
                        def thermal_balance_residual(T):
                            aH = HI_alphaB(T)
                            aHeI = HeI_alphaB(T)
                            aHeII = HeII_alphaB(T)
                            
                            # 冷却項1: 再結合冷却（各イオン種 × n_e × α × kT）
                            cool_recomb = (aH * electron_number_density * HII_number_density +
                                           aHeI * electron_number_density * HeII_number_density +
                                           aHeII * electron_number_density * HeIII_number_density) * k_B * T
                            
                            # 冷却項2: 制動放射冷却 (Z²=1 for H⁺,He⁺; Z²=4 for He²⁺)
                            cool_ff = (Lambda_ff_0 * g_ff(T) * T**0.5 * electron_number_density *
                                       (HII_number_density + HeII_number_density + 4.0 * HeIII_number_density))
                            
                            # 冷却項3: Ly-α 衝突励起冷却
                            cool_lya = electron_number_density * HI_number_density * 7.3e-19 * numpy.exp(-118348.0 / T)
                            
                            return cool_recomb + cool_ff + cool_lya - H_total_vol
                        
                        try:
                            T_new = scipy.optimize.brentq(thermal_balance_residual, 1.0, 1000000.0)
                        except ValueError:
                            if thermal_balance_residual(1.0) > 0:
                                T_new = 1.0
                            else:
                                T_new = 1000000.0
                        
                        # --- 水素とヘリウムの電離平衡（汎用ソルバを使用） ---
                        aH_new = HI_alphaB(T_new)
                        x_H = self._solve_ionization_balance([Gamma_HI], [aH_new], electron_number_density)
                        HI_fraction_new = x_H[0]
                        HII_fraction_new = x_H[1]
                        
                        aHeI_new = HeI_alphaB(T_new)
                        aHeII_new = HeII_alphaB(T_new)
                        x_He = self._solve_ionization_balance(
                            [Gamma_HeI, Gamma_HeII], 
                            [aHeI_new, aHeII_new], 
                            electron_number_density
                        )
                        HeI_fraction_new = x_He[0]
                        HeII_fraction_new = x_He[1]
                        HeIII_fraction_new = x_He[2]
                    
                    # 収束判定
                    if (abs(HII_fraction_new - HII_fraction_loc) < 1e-4 and
                        abs(HeII_fraction_new - HeII_fraction_loc) < 1e-4 and
                        abs(HeIII_fraction_new - HeIII_fraction_loc) < 1e-4 and
                        abs(T_new - T_loc) < 1.0):
                        HII_fraction_loc = HII_fraction_new
                        HeII_fraction_loc = HeII_fraction_new
                        HeIII_fraction_loc = HeIII_fraction_new
                        HeI_fraction_loc = HeI_fraction_new
                        T_loc = T_new
                        break
                    
                    # 局所的な振動（発散）を防ぐため、アンダーリラクゼーション（旧値と新値のブレンド）を適用
                    weight = 0.5
                    HII_fraction_loc = (1.0 - weight) * HII_fraction_loc + weight * HII_fraction_new
                    HeII_fraction_loc = (1.0 - weight) * HeII_fraction_loc + weight * HeII_fraction_new
                    HeIII_fraction_loc = (1.0 - weight) * HeIII_fraction_loc + weight * HeIII_fraction_new
                    HeI_fraction_loc = (1.0 - weight) * HeI_fraction_loc + weight * HeI_fraction_new
                    T_loc = (1.0 - weight) * T_loc + weight * T_new
                
                self.HII_fraction_array[i]      = HII_fraction_loc
                self.HI_fraction_array[i]       = max(1.0 - HII_fraction_loc, 0.0)
                self.HeII_fraction_array[i]     = HeII_fraction_loc
                self.HeIII_fraction_array[i]    = HeIII_fraction_loc
                self.HeI_fraction_array[i]      = HeI_fraction_loc
                self.temperature_array[i]       = T_loc
                
                # 収束した状態でパケットの重みを全光学的厚みで減衰させる
                n_HI_f = self.hydrogen_number_density * self.HI_fraction_array[i]
                n_HeI_f = self.helium_number_density * self.HeI_fraction_array[i]
                n_HeII_f = self.helium_number_density * self.HeII_fraction_array[i]
                dtau_final = (n_HI_f * HI_sigma + n_HeI_f * HeI_sigma + n_HeII_f * HeII_sigma) * dr
                W[active] *= numpy.exp(-dtau_final)
                active = W > 1e-10
            
            max_diff_H = numpy.max(numpy.abs(self.HI_fraction_array - previous_HI_fraction))
            max_diff_He = numpy.max(numpy.abs(self.HeII_fraction_array - previous_HeII_fraction))
            max_diff = max(max_diff_H, max_diff_He)
            time_iter = (time.time() - time_start_total) / 60.0
            print(f"Iteration {global_it + 1:2d}/{max_global_iterations}: "
                  f"max change H={max_diff_H:.2e} He={max_diff_He:.2e}  "
                  f"(経過時間: {time_iter:.3e} min)")
            

        # 自己無撞着なストロームグレン半径の再計算
        # HII領域（x > 0.5）内の体積で重み付けした平均温度を使用
        ionized_mask = self.HII_fraction_array > 0.5
        if numpy.any(ionized_mask):
            volumes = 4 * numpy.pi * self.r_cm**2
            T_mean_HII = numpy.average(self.temperature_array[ionized_mask], weights=volumes[ionized_mask])
        else:
            T_mean_HII = self.T_gas_init
        integrand_Q = numpy.pi * self.B_nu_grid / (h * self.nu_grid)
        Q_H0 = 4 * numpy.pi * self.R_star_cm**2 * numpy.trapezoid(integrand_Q, self.nu_grid)
        n_e_fully_ionized = self.hydrogen_number_density * (1.0 + Y_He)
        self.R_s_cm = (3 * Q_H0 / (4 * numpy.pi * self.hydrogen_number_density * n_e_fully_ionized * HI_alphaB(T_mean_HII)))**(1/3)
        self.R_s_pc = self.R_s_cm / 3.086e18
        print(f"自己無撞着なストロームグレン半径 R_s = {self.R_s_pc:.2f} pc ")
        print(f"(HII領域平均温度 T = {T_mean_HII:.0f} K)")


    def save_tsv(self, filename="simulation.tsv"):
        """計算結果をTSVファイルとして保存する"""
        filepath    = os.path.join(DATA_DIR, filename)        
        data        = numpy.column_stack((self.r_pc, self.HII_fraction_array, self.HI_fraction_array, self.HeIII_fraction_array, self.HeII_fraction_array, self.HeI_fraction_array, self.temperature_array))
        header      = "Radius\tHII_fraction\tHI_fraction\tHeIII_fraction\tHeII_fraction\tHeI_fraction\tTemperature"
        numpy.savetxt(filepath, data, delimiter="\t", header=header, comments="")