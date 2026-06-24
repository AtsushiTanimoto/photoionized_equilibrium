import matplotlib.pyplot
import numpy
import os
import scipy.integrate
import scipy.optimize
import seaborn


# --- 物理定数 (cgs単位系) ---
h = 6.62607015e-27  # プランク定数 [erg s]
c = 2.99792458e10   # 光速度 [cm s^-1]
k_B = 1.380649e-16  # ボルツマン定数 [erg K^-1]
nu_0 = 3.288e15     # 水素の電離限界振動数 (13.6 eV) [Hz]
a_0 = 6.304e-18     # 水素の光電離断面積（電離限界での値）[cm^2]


def alpha_B(T_gas):
    """
    Case B 水素の再結合係数 [cm^3 s^-1]
    Osterbrock & Ferland (2006) の近似式を使用
    """
    return 2.59e-13 * (T_gas / 1e4)**(-0.84)


def a_nu(nu):
    """
    水素の光電離断面積 [cm^2]
    nu > nu_0 において (nu_0 / nu)^3 に比例すると近似
    """
    return numpy.where(nu < nu_0, 0.0, a_0 * (nu_0 / nu)**3)


def B_nu(nu, T_BB):
    """
    プランク関数（黒体放射） [erg s^-1 cm^-2 Hz^-1 sr^-1]
    """
    exponent = h * nu / (k_B * T_BB)
    exponent = numpy.minimum(exponent, 700.0) # オーバーフロー対策
    return (2 * h * nu**3 / c**2) / (numpy.exp(exponent) - 1.0)


# figureディレクトリのパス（プロジェクトルート直下）
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figure')


class PhotoionizationCalculator:
    """光電離平衡を計算するクラス"""
    def __init__(self, n_H=1e2, T_gas=1e4, T_BB=4e4, R_star_solar=10.0, N_nu=200, N_r=2000):
        self.n_H = n_H          # 全水素数密度 [cm^-3]
        self.T_gas_init = T_gas # ガスの初期温度 [K]
        self.T_BB = T_BB        # 星の黒体放射温度 [K]
        self.R_star_cm = R_star_solar * 6.96e10  # 星の半径 [cm]
        self.N_nu = N_nu        # 振動数グリッドの分割数
        self.N_r = N_r          # 距離グリッドの分割数
        
        self.nu_grid = None
        self.a_nu_grid = None
        self.B_nu_grid = None
        self.R_s_cm = None
        self.R_s_pc = None
        self.r_pc = None
        self.r_cm = None
        self.fractions = None
        self.T_gas = None

    def setup_grids(self):
        """振動数と距離のグリッド、および初期定数を設定する"""
        # 連続スペクトルの積分を離散化するため、振動数のグリッドを作成
        # 対数スケールで nu_0 から 100*nu_0 まで分割
        self.nu_grid = numpy.logspace(numpy.log10(nu_0), numpy.log10(100 * nu_0), self.N_nu)
        
        self.a_nu_grid = a_nu(self.nu_grid)
        self.B_nu_grid = B_nu(self.nu_grid, self.T_BB)
        
        # 理論的なストロームグレン半径 R_s を事前計算 (比較および距離グリッド設定用)
        integrand_Q = numpy.pi * self.B_nu_grid / (h * self.nu_grid)
        Q_H0 = 4 * numpy.pi * self.R_star_cm**2 * numpy.trapezoid(integrand_Q, self.nu_grid)
        self.R_s_cm = (3 * Q_H0 / (4 * numpy.pi * self.n_H**2 * alpha_B(self.T_gas_init)))**(1/3)
        self.R_s_pc = self.R_s_cm / 3.086e18
        print(f"理論的なストロームグレン半径 R_s = {self.R_s_pc:.2f} pc")
        
        # 距離グリッドの設定
        # HII領域の境界付近で急激な変化を捉えるため、理論値の 1.5倍 までの距離を分割
        r_pc_max = self.R_s_pc * 1.5
        self.r_pc = numpy.linspace(0.001, r_pc_max, self.N_r)
        self.r_cm = self.r_pc * 3.086e18

    def calculate_equilibrium(self, max_global_iterations=30, tolerance=1e-3, N_packets=200000):
        """モンテカルロ法を利用して光電離平衡の計算を行う（3次元拡張に向けたTop-Down型連続吸収エスティメータ法）"""
        # --- 重要: Optically Thick Trap（Lambda Iteration不安定性）を防ぐため ---
        # 完全に電離した状態 (x=1) からスタートし、外側に向かって中性化させていく「Top-Down」アプローチをとります。
        self.fractions = numpy.ones(self.N_r)
        self.T_gas = numpy.full(self.N_r, float(self.T_gas_init))
        
        print(f"計算を開始します（グローバル反復法: {N_packets}パケット/反復）...")
        
        integrand_Q = numpy.pi * self.B_nu_grid / (h * self.nu_grid)
        Q_cum = scipy.integrate.cumulative_trapezoid(integrand_Q, self.nu_grid, initial=0)
        Q_tot = 4 * numpy.pi * self.R_star_cm**2 * Q_cum[-1]
        
        if Q_cum[-1] > 0:
            CDF = Q_cum / Q_cum[-1]
        else:
            CDF = numpy.linspace(0, 1, len(self.nu_grid))
            
        dr = self.r_cm[1] - self.r_cm[0] if self.N_r > 1 else 0
        V = 4 * numpy.pi * self.r_cm**2 * dr
        Q_p = Q_tot / N_packets
        
        for global_it in range(max_global_iterations):
            old_fractions = self.fractions.copy()
            G_accum = numpy.zeros(self.N_r)
            
            U_nu = numpy.random.rand(N_packets)
            nu_packets = numpy.interp(U_nu, CDF, self.nu_grid)
            a_nu_packets = a_nu(nu_packets)
            
            # 連続吸収法 (Continuous Absorption) を使用するため、パケットは消滅せず重み W が減衰していく
            W = numpy.ones(N_packets)
            active = numpy.ones(N_packets, dtype=bool)
            
            for i in range(self.N_r):
                if not numpy.any(active):
                    self.fractions[i] = 0.0
                    self.T_gas[i] = 10.0
                    continue
                    
                # 局所反復によりセル内の電離度・温度と光学的厚みを整合させる
                x_local = self.fractions[i]
                T_local = self.T_gas[i]
                
                base_dtau = a_nu_packets[active] * dr
                base_factor = Q_p * W[active] / V[i]
                E_heat = h * (nu_packets[active] - nu_0)
                
                for _ in range(10): 
                    n_H0 = self.n_H * (1.0 - x_local)
                    dtau = n_H0 * base_dtau
                    
                    # 解析的パスレングス・エスティメータ
                    if n_H0 > 1e-10:
                        factor = base_factor * (1.0 - numpy.exp(-dtau)) / n_H0
                    else:
                        factor = base_factor * base_dtau
                        
                    G = numpy.sum(factor)
                    H_total = numpy.sum(factor * E_heat)
                    G_accum[i] = G
                    
                    if G <= 0:
                        x_new = 0.0
                        T_new = 10.0
                    else:
                        def thermal_balance_residual(T):
                            term1 = G * k_B * T
                            term2 = G * 1.42e-27 * T**0.5 / alpha_B(T)
                            term3 = x_local * self.n_H * 7.3e-19 * numpy.exp(-118348.0 / T)
                            return term1 + term2 + term3 - H_total
                            
                        try:
                            T_new = scipy.optimize.brentq(thermal_balance_residual, 1.0, 1000000.0)
                        except ValueError:
                            if thermal_balance_residual(1.0) > 0:
                                T_new = 1.0
                            else:
                                T_new = 1000000.0
                                
                        alpha_new = alpha_B(T_new)
                        x_new = (2 * G) / (G + numpy.sqrt(G**2 + 4 * self.n_H * alpha_new * G))
                    
                    # 局所更新と収束判定
                    if abs(x_new - x_local) < 1e-4 and abs(T_new - T_local) < 1.0:
                        x_local = x_new
                        T_local = T_new
                        break
                        
                    x_local = x_new
                    T_local = T_new
                
                self.fractions[i] = x_local
                self.T_gas[i] = T_local
                
                # 収束した状態でパケットの重みを減衰させる
                n_H0_final = self.n_H * (1.0 - x_local)
                dtau_final = n_H0_final * a_nu_packets[active] * dr
                W[active] *= numpy.exp(-dtau_final)
                active = W > 1e-10
                
            max_diff = numpy.max(numpy.abs(self.fractions - old_fractions))
            print(f"Iteration {global_it + 1:2d}/{max_global_iterations}: max fraction change = {max_diff:.2e}")
            
            if max_diff < tolerance:
                print(f"計算が完了しました（{global_it + 1}回のグローバル反復で収束）。")
                break
        else:
            print(f"警告: 最大反復回数 ({max_global_iterations}) に到達しましたが、完全には収束していません。")

    def plot_results(self):
        """計算結果をプロットして保存する（イオン割合と温度を別々の図として出力）"""
        os.makedirs(FIGURE_DIR, exist_ok=True)
        seaborn.set_theme()
        seaborn.set_style("whitegrid")

        # --- イオン割合の図 ---
        fig1, ax1 = matplotlib.pyplot.subplots(figsize=(16, 9))
        ax1.plot(self.r_pc, self.fractions, color='blue', linewidth=2, label='Numerical Result')
        ax1.axvline(self.R_s_pc, color='red', linestyle='--', label=f'Theoretical $R_s$ = {self.R_s_pc:.2f} pc')
        ax1.set_xlabel('Distance from Star [pc]')
        ax1.set_ylabel('Ionization Fraction $x$')
        ax1.set_title(f'Ionization Fraction\n$T_{{BB}}={self.T_BB}$K, $n_H={self.n_H}$ cm$^{{-3}}$')
        ax1.legend(loc="lower left")
        ax1.set_ylim([0, 1.05])
        ax1.set_xlim([0, self.r_pc[-1]])
        matplotlib.pyplot.tight_layout()
        ionization_filename = os.path.join(FIGURE_DIR, 'ionization_fraction.png')
        matplotlib.pyplot.savefig(ionization_filename, dpi=150)
        matplotlib.pyplot.close(fig1)
        print(f"イオン割合のグラフを '{ionization_filename}' に保存しました。")

        # --- 温度の図 ---
        fig2, ax2 = matplotlib.pyplot.subplots(figsize=(16, 9))
        ax2.plot(self.r_pc, self.T_gas, color='orange', linewidth=2, label='Gas Temperature')
        ax2.set_xlabel('Distance from Star [pc]')
        ax2.set_ylabel('Temperature [K]')
        ax2.set_title(f'Gas Temperature\n$T_{{BB}}={self.T_BB}$K, $n_H={self.n_H}$ cm$^{{-3}}$')
        ax2.legend(loc="upper right")
        ax2.set_xlim([0, self.r_pc[-1]])
        matplotlib.pyplot.tight_layout()
        temperature_filename = os.path.join(FIGURE_DIR, 'temperature.png')
        matplotlib.pyplot.savefig(temperature_filename, dpi=150)
        matplotlib.pyplot.close(fig2)
        print(f"温度のグラフを '{temperature_filename}' に保存しました。")

        print(f"最も内側 (r = {self.r_pc[0]:.4f} pc) での電離度: {self.fractions[0]:.6f}, 温度: {self.T_gas[0]:.1f} K")
        print(f"最も外側 (r = {self.r_pc[-1]:.4f} pc) での電離度: {self.fractions[-1]:.6e}, 温度: {self.T_gas[-1]:.1f} K")


if __name__ == "__main__":
    # --- 計算用パラメータ設定 ---
    # デフォルトのパラメータで計算器を初期化
    calculator = PhotoionizationCalculator(
        n_H=1e4,          # 全水素数密度 [cm^-3] (例: 星間雲)
        T_gas=1e4,        # ガスの温度 [K] (通常HII領域では約1万度)
        T_BB=4e4,         # 星の黒体放射温度 [K] (例: 高温なO型星)
        R_star_solar=10,  # 星の半径 (太陽半径の10倍と仮定)
        N_nu=200,         # 振動数分割数
        N_r=2000          # 距離分割数
    )
    
    # 計算の実行
    calculator.setup_grids()
    calculator.calculate_equilibrium(N_packets=1000000, max_global_iterations=100)
    calculator.plot_results()
