"""
光電離平衡計算のメインスクリプト
"""
import astropy.constants
import astropy.units
import numpy
from calculator import PhotoionizationCalculator
from figure import plot_results


if __name__ == "__main__":
    radius_grid_array       = numpy.linspace(0.000e+00, 1.000e+00, 101)*astropy.constants.pc.to(astropy.units.cm)
    hydrogen_number_density = 1.000e+04

    # デフォルトのパラメータで計算器を初期化
    calculator = PhotoionizationCalculator(radius_grid_array=radius_grid_array, hydrogen_number_density=hydrogen_number_density,
        T_gas=1e4,        # ガスの温度 [K] (通常HII領域では約1万度)
        T_BB=4e4,         # 星の黒体放射温度 [K] (例: 高温なO型星)
        R_star_solar=10.0,  # 星の半径 (太陽半径の10倍と仮定)
        N_nu=200,         # 振動数分割数
    )
    
    # 計算の実行
    calculator.setup_grids()
    calculator.calculate_equilibrium(N_packets=1000000, max_global_iterations=100)
    plot_results(calculator)
    calculator.save_tsv()
