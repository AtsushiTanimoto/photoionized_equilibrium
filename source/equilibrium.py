"""
光電離平衡を計算する PhotoionizationModel クラスの定義
"""
import numpy
import os
import scipy.integrate
import scipy.optimize
import time

from constants import h, k_B, H01_nu0, He01_nu0, He02_nu0, Lambda_ff_0, Y_He
from physics import (g_ff, H01_alphaB, He01_alphaB, He02_alphaB, H01_anu, He01_anu, He02_anu, B_nu)


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
        self.N_r                        = None
        self.r_cm                       = None
        self.r_pc                       = None
        self.H00_fraction_array         = None
        self.H01_fraction_array         = None
        self.He00_fraction_array        = None        # He⁺ 割合 x_HeII = n(He⁺) / n_He
        self.He01_fraction_array        = None       # He²⁺ 割合 x_HeIII = n(He²⁺) / n_He
        self.He02_fraction_array        = None
        self.T_gas                      = None


    def setup_grids(self):
        """振動数と距離のグリッド、および初期定数を設定する"""
        # radius_grid_array [cm] から距離グリッドを設定
        self.r_cm = numpy.asarray(self.radius_grid_array, dtype=float)
        self.N_r = len(self.r_cm)
        self.r_pc = self.r_cm / 3.086e18

        # 連続スペクトルの積分を離散化するため、振動数のグリッドを作成
        # 対数スケールで nu_0 から 1000*nu_0 まで分割
        self.nu_grid = numpy.linspace(H01_nu0, 1.000e+03*H01_nu0, self.N_nu)
        self.a_nu_grid = H01_anu(self.nu_grid)
        self.B_nu_grid = B_nu(self.nu_grid, self.T_BB)
        
        # 理論的なストロームグレン半径 R_s を事前計算 (比較用)
        # ヘリウムによる電子密度の増加を考慮: n_e ≈ n_H(1 + Y_He) (完全電離時)
        integrand_Q         = numpy.pi * self.B_nu_grid / (h * self.nu_grid)
        Q_H0                = 4 * numpy.pi * self.R_star_cm**2 * numpy.trapezoid(integrand_Q, self.nu_grid)
        n_e_fully_ionized   = self.hydrogen_number_density * (1.0 + Y_He)
        self.R_s_cm         = (3 * Q_H0 / (4 * numpy.pi * self.hydrogen_number_density * n_e_fully_ionized * H01_alphaB(self.T_gas_init)))**(1/3)
        self.R_s_pc         = self.R_s_cm / 3.086e18
        print(f"理論的なストロームグレン半径 R_s = {self.R_s_pc:.2f} pc")

    def calculate_equilibrium(self, max_global_iterations=30, tolerance=1e-3, N_packets=200000):
        """
        モンテカルロ法を利用して光電離平衡の計算を行う（水素 + ヘリウム）

        光子パケットは H I, He I, He II の3つの吸収体と相互作用する。
        各セルで光学的厚みは全吸収体の和 τ = (n_HI σ_HI + n_HeI σ_HeI + n_HeII σ_HeII) dr で、
        各種への配分は不透明度の寄与比で決まる。
        """
        # --- Top-Down approach: 完全電離状態からスタート ---
        self.H00_fraction_array     = numpy.zeros(self.N_r)         # x_H = 1 (全て H⁺)
        self.H01_fraction_array     = numpy.ones(self.N_r)
        self.He00_fraction_array    = numpy.zeros(self.N_r)
        self.He01_fraction_array    = numpy.zeros(self.N_r)
        self.He02_fraction_array    = numpy.ones(self.N_r)
        self.T_gas = numpy.full(self.N_r, float(self.T_gas_init))
        
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
            previous_H01_fraction = self.H01_fraction_array.copy()
            previous_He01_fraction = self.He01_fraction_array.copy()
            
            U_nu = numpy.random.rand(N_packets)
            nu_packets = numpy.interp(U_nu, CDF, self.nu_grid)
            
            # 各パケットの光電離断面積を事前計算
            H01_sigma_all   = H01_anu(nu_packets)
            He01_sigma_all  = He01_anu(nu_packets)
            He02_sigma_all  = He02_anu(nu_packets)
            
            # 各吸収体による余剰エネルギー（加熱に寄与）
            H01_E_heat_all  = h * numpy.maximum(nu_packets - H01_nu0, 0.0)
            He01_E_heat_all = h * numpy.maximum(nu_packets - He01_nu0, 0.0)
            He02_E_heat_all = h * numpy.maximum(nu_packets - He02_nu0, 0.0)
            
            # 連続吸収法: パケットは消滅せず重み W が減衰していく
            W = numpy.ones(N_packets)
            active = numpy.ones(N_packets, dtype=bool)
            
            for i in range(self.N_r):
                if not numpy.any(active):
                    self.H00_fraction_array[i] = 0.0
                    self.H01_fraction_array[i] = 1.0
                    self.He00_fraction_array[i] = 0.0
                    self.He01_fraction_array[i] = 0.0
                    self.He02_fraction_array[i] = 1.0
                    self.T_gas[i] = 10.0
                    continue
                
                # 局所反復: セル内の電離度・温度と光学的厚みを整合させる
                H00_fraction_loc    = self.H00_fraction_array[i]
                He00_fraction_loc   = self.He00_fraction_array[i]
                He01_fraction_loc   = self.He01_fraction_array[i]
                He02_fraction_loc   = self.He02_fraction_array[i]
                T_loc               = self.T_gas[i]
                H01_sigma           = H01_sigma_all[active]
                He01_sigma          = He01_sigma_all[active]
                He02_sigma          = He02_sigma_all[active]
                H01_E_heat          = H01_E_heat_all[active]
                He01_E_heat         = He01_E_heat_all[active]
                He02_E_heat         = He02_E_heat_all[active]
                base_factor         = Q_p * W[active] / V[i]
                
                for _ in range(20):  # 連立方程式なので反復回数を増やす
                    # 数密度の計算
                    H00_number_density      = self.hydrogen_number_density * H00_fraction_loc
                    H01_number_density      = self.hydrogen_number_density * max(1.0 - H00_fraction_loc, 1e-15)
                    He00_number_density     = self.helium_number_density * He00_fraction_loc
                    He01_number_density     = self.helium_number_density * He01_fraction_loc
                    He02_number_density     = self.helium_number_density * He02_fraction_loc
                    electron_number_density = H00_number_density + He01_number_density + 2.0 * He00_number_density
                    
                    # 全光学的厚み τ = κ × dr
                    kappa = H01_number_density * H01_sigma + He02_number_density * He02_sigma + He01_number_density * He01_sigma
                    tau_total = kappa * dr
                    
                    # φ(τ) = (1 - e^{-τ}) / τ: τ→0 で 1 に漸近
                    # 解析的パスレングス・エスティメータの多種拡張
                    phi = numpy.where(
                        tau_total > 1e-8,
                        (1.0 - numpy.exp(-tau_total)) / tau_total,
                        1.0 - tau_total / 2.0)
                    
                    # 各種の光電離率（1原子あたり） [s⁻¹]
                    common = base_factor * phi * dr
                    Gamma_HI = numpy.sum(common * H01_sigma)
                    Gamma_HeI = numpy.sum(common * He02_sigma)
                    Gamma_HeII = numpy.sum(common * He01_sigma)
                    
                    # 加熱率（体積あたり） [erg s⁻¹ cm⁻³]
                    # H_vol = Σ_s n_s × Σ_packets (common × σ_s × E_s)
                    H_total_vol = (
                        H01_number_density * numpy.sum(common * H01_sigma * H01_E_heat) +
                        He02_number_density * numpy.sum(common * He02_sigma * He02_E_heat) +
                        He01_number_density * numpy.sum(common * He01_sigma * He01_E_heat))
                    
                    if Gamma_HI + Gamma_HeI + Gamma_HeII <= 0:
                        H00_fraction_new = 0.0
                        He01_fraction_new = 0.0
                        He00_fraction_new = 0.0
                        T_new = 10.0
                    else:
                        # --- 温度を熱収支方程式から求解 ---
                        def thermal_balance_residual(T):
                            aH = H01_alphaB(T)
                            aHeI = He02_alphaB(T)
                            aHeII = He01_alphaB(T)
                            
                            # 冷却項1: 再結合冷却（各イオン種 × n_e × α × kT）
                            cool_recomb = (aH * electron_number_density * H00_number_density +
                                           aHeI * electron_number_density * He01_number_density +
                                           aHeII * electron_number_density * He00_number_density) * k_B * T
                            
                            # 冷却項2: 制動放射冷却 (Z²=1 for H⁺,He⁺; Z²=4 for He²⁺)
                            cool_ff = (Lambda_ff_0 * g_ff(T) * T**0.5 * electron_number_density *
                                       (H00_number_density + He01_number_density + 4.0 * He00_number_density))
                            
                            # 冷却項3: Ly-α 衝突励起冷却
                            cool_lya = electron_number_density * H01_number_density * 7.3e-19 * numpy.exp(-118348.0 / T)
                            
                            return cool_recomb + cool_ff + cool_lya - H_total_vol
                        
                        try:
                            T_new = scipy.optimize.brentq(thermal_balance_residual, 1.0, 1000000.0)
                        except ValueError:
                            if thermal_balance_residual(1.0) > 0:
                                T_new = 1.0
                            else:
                                T_new = 1000000.0
                        
                        # --- 水素の電離平衡 ---
                        # Γ_HI × (1-x_H) = α_B × n_e × x_H
                        #   → x_H = Γ_HI / (Γ_HI + α_B × n_e)
                        aH_new = H01_alphaB(T_new)
                        if Gamma_HI > 0 and electron_number_density > 0:
                            H00_fraction_new = Gamma_HI / (Gamma_HI + aH_new * electron_number_density)
                        elif Gamma_HI > 0:
                            H00_fraction_new = 1.0
                        else:
                            H00_fraction_new = 0.0
                        
                        # --- ヘリウムの電離平衡 ---
                        # He I: Γ_HeI × n_HeI = α_HeI × n_e × n_HeII
                        #   → r1 = n_HeI/n_HeII = α_HeI × n_e / Γ_HeI
                        # He II: Γ_HeII × n_HeII = α_HeII × n_e × n_HeIII
                        #   → r2 = n_HeIII/n_HeII = Γ_HeII / (α_HeII × n_e)
                        # 規格化: x_HeI + x_HeII + x_HeIII = 1
                        #   → x_HeII = 1 / (r1 + 1 + r2)
                        aHeI_new = He02_alphaB(T_new)
                        aHeII_new = He01_alphaB(T_new)
                        
                        if Gamma_HeI > 1e-30:
                            r1 = aHeI_new * electron_number_density / Gamma_HeI
                        else:
                            r1 = 1e30  # ヘリウムは中性のまま
                        
                        if aHeII_new * electron_number_density > 1e-30:
                            r2 = Gamma_HeII / (aHeII_new * electron_number_density)
                        else:
                            r2 = 0.0  # He²⁺ は生成されない
                        
                        denom = r1 + 1.0 + r2
                        He01_fraction_new = 1.0 / denom
                        He00_fraction_new = r2 / denom
                    
                    He02_fraction_new = max(1.0 - He01_fraction_new - He00_fraction_new, 0.0)
                    
                    # 収束判定
                    if (abs(H00_fraction_new - H00_fraction_loc) < 1e-4 and
                        abs(He01_fraction_new - He01_fraction_loc) < 1e-4 and
                        abs(He00_fraction_new - He00_fraction_loc) < 1e-4 and
                        abs(T_new - T_loc) < 1.0):
                        H00_fraction_loc = H00_fraction_new
                        He01_fraction_loc = He01_fraction_new
                        He00_fraction_loc = He00_fraction_new
                        He02_fraction_loc = He02_fraction_new
                        T_loc = T_new
                        break
                    
                    H00_fraction_loc = H00_fraction_new
                    He01_fraction_loc = He01_fraction_new
                    He00_fraction_loc = He00_fraction_new
                    He02_fraction_loc = He02_fraction_new
                    T_loc = T_new
                
                self.H00_fraction_array[i] = H00_fraction_loc
                self.H01_fraction_array[i] = max(1.0 - H00_fraction_loc, 0.0)
                self.He01_fraction_array[i] = He01_fraction_loc
                self.He00_fraction_array[i] = He00_fraction_loc
                self.He02_fraction_array[i] = He02_fraction_loc
                self.T_gas[i] = T_loc
                
                # 収束した状態でパケットの重みを全光学的厚みで減衰させる
                n_HI_f = self.hydrogen_number_density * self.H01_fraction_array[i]
                n_HeI_f = self.helium_number_density * self.He02_fraction_array[i]
                n_HeII_f = self.helium_number_density * self.He01_fraction_array[i]
                dtau_final = (n_HI_f * H01_sigma + n_HeI_f * He02_sigma + n_HeII_f * He01_sigma) * dr
                W[active] *= numpy.exp(-dtau_final)
                active = W > 1e-10
            
            max_diff_H = numpy.max(numpy.abs(self.H01_fraction_array - previous_H01_fraction))
            max_diff_He = numpy.max(numpy.abs(self.He01_fraction_array - previous_He01_fraction))
            max_diff = max(max_diff_H, max_diff_He)
            time_iter = (time.time() - time_start_total) / 60.0
            print(f"Iteration {global_it + 1:2d}/{max_global_iterations}: "
                  f"max change H={max_diff_H:.2e} He={max_diff_He:.2e}  "
                  f"(経過時間: {time_iter:.3e} min)")
            
            if max_diff < tolerance:
                time_total = (time.time() - time_start_total) / 60.0
                print(f"計算が完了しました（{global_it + 1}回のグローバル反復で収束, "
                      f"計算時間: {time_total:.3e} min）。")
                break
        else:
            time_total = (time.time() - time_start_total) / 60.0
            print(f"警告: 最大反復回数 ({max_global_iterations}) に到達しましたが、"
                  f"完全には収束していません（計算時間: {time_total:.3e} min）。")

        # 自己無撞着なストロームグレン半径の再計算
        # HII領域（x > 0.5）内の平均温度を使用
        ionized_mask = self.H00_fraction_array > 0.5
        if numpy.any(ionized_mask):
            T_mean_HII = numpy.mean(self.T_gas[ionized_mask])
        else:
            T_mean_HII = self.T_gas_init
        integrand_Q = numpy.pi * self.B_nu_grid / (h * self.nu_grid)
        Q_H0 = 4 * numpy.pi * self.R_star_cm**2 * numpy.trapezoid(integrand_Q, self.nu_grid)
        n_e_fully_ionized = self.hydrogen_number_density * (1.0 + Y_He)
        self.R_s_cm = (3 * Q_H0 / (4 * numpy.pi * self.hydrogen_number_density * n_e_fully_ionized * H01_alphaB(T_mean_HII)))**(1/3)
        self.R_s_pc = self.R_s_cm / 3.086e18
        print(f"自己無撞着なストロームグレン半径 R_s = {self.R_s_pc:.2f} pc ")
        print(f"(HII領域平均温度 T = {T_mean_HII:.0f} K)")


    def save_tsv(self, filename="simulation.tsv"):
        """計算結果をTSVファイルとして保存する"""
        filepath    = os.path.join(DATA_DIR, filename)        
        data        = numpy.column_stack((self.r_pc, self.H00_fraction_array, self.H01_fraction_array, self.He00_fraction_array, self.He01_fraction_array, self.He02_fraction_array, self.T_gas))
        header      = "Radius\tH00_fraction\tH01_fraction\tHe00_fraction\tHe01_fraction\tHe02_fraction\tTemperature"
        numpy.savetxt(filepath, data, delimiter="\t", header=header, comments="")