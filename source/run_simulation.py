"""
光電離平衡計算のメインスクリプト
"""
import astropy.constants
import astropy.units
import numpy
from equilibrium import PhotoionizationModel
from visualization import plot_results


if __name__ == "__main__":
    radius_grid_array       = numpy.linspace(0.000e+00, 1.000e+00, 101)*astropy.constants.pc.to(astropy.units.cm)
    hydrogen_number_density = 1.000e+04
    number_of_energy        = 1001

    # デフォルトのパラメータで計算器を初期化
    model = PhotoionizationModel(radius_grid_array=radius_grid_array, hydrogen_number_density=hydrogen_number_density,
        T_gas=1e4,        # ガスの温度 [K] (通常HII領域では約1万度)
        T_BB=4e4,         # 星の黒体放射温度 [K] (例: 高温なO型星)
        R_star_solar=10.0,  # 星の半径 (太陽半径の10倍と仮定)
        N_nu=number_of_energy,         # 振動数分割数
    )
    
    # 計算の実行
    model.setup_grids()
    model.calculate_equilibrium(N_packets=2000000, max_global_iterations=1)
    plot_results(model)
    model.save_tsv()
