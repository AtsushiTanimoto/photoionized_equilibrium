"""
FAC の生データ (pfac.atom.atomic_data が出力する <元素><電子数 2 桁>a.rr) から、基底準位の光電離断面積の
フィットパラメータを共有データベースと同じ .pi 形式で書き出す。共有データベース (ATOMIC_DATA_DIR) にない
電子数 > 10 のイオン用で、出力は constants.FAC_SUPPLEMENT_DIR/<元素>/<元素><電子数>.pi。

  python3 make_pi_from_fac.py <元素記号> <電子数> <FAC の出力ディレクトリ>
  例: python3 make_pi_from_fac.py Mg 12 /path/to/fac_mg/Mg

FAC の生データは例えば
  python3 -c "import pfac.atom; pfac.atom.atomic_data(nele=[11, 12], asym='Mg', dir='Mg/')"
で作る (~/github/atomicdatabase/source/atomic_data.py と同じ呼び出し)。

.rr の形式: 遷移ごとに見出し行「束縛準位 2J 電離後の準位 2J 閾値 [eV] 軌道 (n*1000+l)」、
次にフィットパラメータの行 (4 数値)、続いてエネルギーグリッドの行「E − I [eV]  (RR 断面積)  光電離断面積 [10⁻²⁰ cm²]  (…)」。
~/github/atomicdatabase/source/photoionization_data.py と同じく、最初の 1 行 (パラメータ) を捨てた 6 点を
σ(E) = Σ0 (E/I)^Γ exp(−E/τ) でフィットする。共有データベースの既存ファイル (Γ もフィット) に合わせて Γ も自由にし
（対数を取って線形最小二乗）、τ が負になったら Γ = −3 に固定する。
"""
import os
import sys

import numpy

import constants


def read_ground_level_channels(rr_path):
    """.rr から基底準位 (束縛準位 0) のチャネルを読む。戻り値: [(見出し列のリスト, E [eV], σ [cm²]), ...]"""
    channels = []
    header, energies, cross_sections = None, [], []
    with open(rr_path) as f:
        for line in f:
            columns = line.split()
            if len(columns) == 6 and all(c.lstrip("-").replace(".", "").replace("E+", "").replace("E-", "").isdigit() for c in columns):
                if header is not None and header[0] == 0:
                    channels.append((header, numpy.array(energies[1:]), numpy.array(cross_sections[1:])))   # 1 点目はパラメータ行
                header = [int(columns[0]), int(columns[1]), int(columns[2]), int(columns[3]), float(columns[4]), int(columns[5])]
                energies, cross_sections = [], []
            elif len(columns) == 4 and header is not None:
                try:
                    energies.append(float(columns[0]) + header[4])
                    cross_sections.append(1.0e-20 * float(columns[2]))
                except ValueError:
                    pass
    if header is not None and header[0] == 0:
        channels.append((header, numpy.array(energies[1:]), numpy.array(cross_sections[1:])))
    return channels


def fit_channel(I, E, sigma):
    """
    σ(E) = Σ0 (E/I)^Γ exp(−E/τ) のフィット。戻り値 (Σ0, Γ, τ)。
    対数を取ると log σ = log Σ0 + Γ log(E/I) − E/τ と線形になるので最小二乗で解く (スケールの問題がなく頑健)。
    τ < 0 (高エネルギーで増加) になったら Γ = −3 に固定して Σ0, τ を解き直す。
    """
    positive = sigma > 0.0
    if numpy.count_nonzero(positive) < 3:
        return None
    E, sigma = E[positive], sigma[positive]
    A = numpy.column_stack([numpy.ones_like(E), numpy.log(E / I), -E])
    coefficients = numpy.linalg.lstsq(A, numpy.log(sigma), rcond=None)[0]
    s0, gamma, inverse_tau = numpy.exp(coefficients[0]), coefficients[1], coefficients[2]
    if inverse_tau <= 0.0:
        A = numpy.column_stack([numpy.ones_like(E), -E])
        coefficients = numpy.linalg.lstsq(A, numpy.log(sigma) + 3.0 * numpy.log(E / I), rcond=None)[0]
        s0, gamma, inverse_tau = numpy.exp(coefficients[0]), -3.0, max(coefficients[1], 1.0e-6)
    return float(s0), float(gamma), float(1.0 / inverse_tau)


def write_pi(element, number_of_electrons, fac_dir):
    rr_path = os.path.join(fac_dir, f"{element}{number_of_electrons:02d}a.rr")
    channels = read_ground_level_channels(rr_path)
    if not channels:
        raise ValueError(f"{rr_path} に基底準位のチャネルがない")
    output_dir = os.path.join(constants.FAC_SUPPLEMENT_DIR, element)
    os.makedirs(output_dir, exist_ok=True)
    output = os.path.join(output_dir, f"{element}{number_of_electrons:02d}.pi")
    N_written = 0
    with open(output, "w") as f:
        for header, E, sigma in channels:
            fit = fit_channel(header[4], E, sigma)
            if fit is None:
                continue
            s0, gamma, tau = fit
            f.write(f"{header[0]:6d} {header[1]:3d}   {header[2]:6d} {header[3]:3d}  {header[5]:6d}      {header[4]:11.5e}   {s0:10.4e}  {gamma:10.4e}   {tau:10.4e}\n")
            N_written += 1
    print(f"{output}: {N_written} channels from {len(channels)} (ground level)")
    return output


if __name__ == "__main__":
    write_pi(sys.argv[1], int(sys.argv[2]), sys.argv[3])
