"""
イオンの衝突励起による輝線冷却 (CHIANTI の準位・遷移・衝突強度データ、全元素共通)

各イオンについて低い方から N_LEVELS_MAX 準位を考え、電子衝突（および C I, C II の微細構造準位では
H 原子衝突）と自発放射の統計平衡から準位占有率を解き、冷却率 Λ = Σ_i n_i Σ_j A_ij ΔE_ij を求める。
計算を速くするため、衝突係数は温度グリッド上で読み込み時に準備しておき、
LineCoolingSet.table(n_e, n_HI, fractions) で温度グリッド上の冷却率の表を作り、
LineCoolingSet.rate(T, table) で内挿して使う。
CHIANTI に衝突強度 (.scups) がないイオン (例: Fe XVI 以下の一部、Cr, Mn の多く) は冷却に寄与しないとみなす。
"""
import numpy
import os
import scipy.interpolate

from constants import h, c, k_B, CHIANTI_DIR


N_LEVELS_MAX = 20       # 各イオンで考慮する準位数の上限（低い方から）。60 準位との差は T ≤ 2×10⁴ K の C II-C IV で 0.1% 以下
Ry = 2.1798723611e-11   # リュードベリ [erg]

# 冷却率の表の温度グリッド [K] (log10 T = 0, ..., 8)
LOG_T_GRID = numpy.linspace(0.0, 8.0, 321)
T_GRID = 10**LOG_T_GRID


def _read_until_terminator(filepath):
    """CHIANTI のファイルの数値部分（最初の "-1" の行まで）を行ごとに返す"""
    lines = []
    with open(filepath) as f:
        for line in f:
            if line.strip() == "-1":
                break
            lines.append(line)
    return lines


def _read_elvlc(filepath):
    """
    準位ファイル (.elvlc) を読み込む。
    戻り値: {準位番号: (統計的重み g, エネルギー [cm^-1])}
    観測値がない準位 (観測エネルギー < 0) は理論値を使う。
    """
    levels = {}
    for line in _read_until_terminator(filepath):
        columns = line.split(",")[0].split()  # C IV などでは "," 以降に余分な列がある
        index = int(columns[0])
        J, energy_observed, energy_theory = float(columns[-3]), float(columns[-2]), float(columns[-1])
        energy = energy_observed if energy_observed >= 0.0 else energy_theory
        levels[index] = (2.0 * J + 1.0, energy)
    return levels


def _read_wgfa(filepath):
    """放射遷移ファイル (.wgfa) を読み込む。戻り値: [(準位1, 準位2, A [s^-1]), ...]"""
    transitions = []
    for line in _read_until_terminator(filepath):
        columns = line.split()
        transitions.append((int(columns[0]), int(columns[1]), float(columns[4])))
    return transitions


def _read_scups(filepath):
    """
    電子衝突強度ファイル (.scups) を読み込む。3 行で 1 遷移:
      下準位, 上準位, ΔE [Ry], gf, 高エネルギー極限, 節点数, 型, スケーリング定数 C
      スケールした温度の節点
      スケールした衝突強度の節点
    戻り値: [(下準位, 上準位, ΔE [Ry], 型, C, 節点 x, 節点 y), ...]
    """
    lines = _read_until_terminator(filepath)
    transitions = []
    for k in range(0, len(lines) - 2, 3):
        header = lines[k].split()
        x = numpy.array(lines[k + 1].split(), dtype=float)
        y = numpy.array(lines[k + 2].split(), dtype=float)
        transitions.append((int(header[0]), int(header[1]), float(header[2]), int(header[6]), float(header[7]), x, y))
    return transitions


def effective_collision_strength(T_gas, delta_E_Ry, transition_type, C, x_knots, y_knots):
    """
    Burgess & Tully (1992) のスケーリングを戻して有効衝突強度 Υ(T) を求める（CHIANTI の規約）
    スケールした衝突強度は 3 次スプラインで内挿する。
    """
    kT_over_dE = k_B * T_gas / (delta_E_Ry * Ry)
    if transition_type in (1, 4):
        x = 1.0 - numpy.log(C) / numpy.log(kT_over_dE + C)
    else:
        x = kT_over_dE / (kT_over_dE + C)
    spline = scipy.interpolate.splrep(x_knots, y_knots, s=0, k=min(3, len(x_knots) - 1))
    y = scipy.interpolate.splev(numpy.clip(x, 0.0, 1.0), spline)
    if transition_type == 1:
        return y * numpy.log(kT_over_dE + numpy.e)
    elif transition_type == 2:
        return y
    elif transition_type == 3:
        return y / (kT_over_dE + 1.0)
    elif transition_type == 4:
        return y * numpy.log(kT_over_dE + C)
    elif transition_type == 5:
        return y / kT_over_dE
    elif transition_type == 6:
        return 10**y
    raise ValueError(f"Unsupported Burgess-Tully transition type {transition_type}")


# H 原子衝突による微細構造準位の脱励起係数 [cm^3 s^-1] (Hollenbach & McKee 1989, Table 8)
# キー: (CHIANTI のイオン名, 上準位, 下準位), 値: T [K] の関数
H_DEEXCITATION_RATE = {
    ("c_1", 2, 1): lambda T: 1.6e-10 * (T / 100.0)**0.14,   # C I  3P1 → 3P0
    ("c_1", 3, 1): lambda T: 9.2e-11 * (T / 100.0)**0.26,   # C I  3P2 → 3P0
    ("c_1", 3, 2): lambda T: 2.9e-10 * (T / 100.0)**0.26,   # C I  3P2 → 3P1
    ("c_2", 2, 1): lambda T: 8.0e-10 * (T / 100.0)**0.07,   # C II 2P3/2 → 2P1/2 (158 μm)
}


def chianti_ion_directory(name):
    """CHIANTI のイオン名 (例: "fe_25") からディレクトリを返す"""
    return os.path.join(CHIANTI_DIR, name.split("_")[0], name)


def line_cooling_data_available(name):
    """輝線冷却の計算に必要な .elvlc, .wgfa, .scups が揃っているか"""
    directory = chianti_ion_directory(name)
    return all(os.path.exists(os.path.join(directory, f"{name}.{ext}")) for ext in ("elvlc", "wgfa", "scups"))


class LineCoolingIon:
    """1 つのイオンの多準位原子モデル (name は CHIANTI のイオン名、例: "c_2", "fe_25")"""
    def __init__(self, name, N_levels_max=N_LEVELS_MAX):
        self.name = name
        directory = chianti_ion_directory(name)
        levels = _read_elvlc(os.path.join(directory, f"{name}.elvlc"))
        N = min(N_levels_max, len(levels))
        self.N_levels = N
        self.g = numpy.array([levels[i + 1][0] for i in range(N)])
        self.energy = numpy.array([levels[i + 1][1] for i in range(N)]) * h * c  # [erg]
        T = T_GRID[:, None]

        # 自発放射: A[i, j] は準位 i → j の遷移確率（エネルギーの高い方から低い方へ）
        self.A = numpy.zeros((N, N))
        for level1, level2, A in _read_wgfa(os.path.join(directory, f"{name}.wgfa")):
            i, j = level1 - 1, level2 - 1
            if i >= N or j >= N or i == j or A <= 0.0:
                continue
            upper, lower = (i, j) if self.energy[i] > self.energy[j] else (j, i)
            self.A[upper, lower] += A
        delta_E = numpy.maximum(self.energy[:, None] - self.energy[None, :], 0.0)
        # 各準位から自発放射で放出されるエネルギー Σ_j A_ij ΔE_ij [erg s^-1]
        self.radiated_power = numpy.sum(self.A * delta_E, axis=1)

        # 電子衝突: C_e[t, i, j] は温度 T_GRID[t] で準位 i → j の係数 [cm^3 s^-1]
        self.C_e = numpy.zeros((len(T_GRID), N, N))
        for level1, level2, delta_E_Ry, transition_type, C, x_knots, y_knots in _read_scups(os.path.join(directory, f"{name}.scups")):
            i, j = level1 - 1, level2 - 1
            if i >= N or j >= N or i == j:
                continue
            lower, upper = (i, j) if self.energy[i] < self.energy[j] else (j, i)
            upsilon = numpy.maximum(effective_collision_strength(T_GRID, delta_E_Ry, transition_type, C, x_knots, y_knots), 0.0)
            q_down = 8.629e-6 / (self.g[upper] * numpy.sqrt(T_GRID)) * upsilon
            self._add_collision_pair(self.C_e, lower, upper, q_down)

        # H 原子衝突: C_H[t, i, j] [cm^3 s^-1]
        self.C_H = numpy.zeros((len(T_GRID), N, N))
        for (ion_name, upper, lower), rate in H_DEEXCITATION_RATE.items():
            if ion_name == name:
                self._add_collision_pair(self.C_H, lower - 1, upper - 1, rate(T_GRID))

    def _add_collision_pair(self, C_matrix, lower, upper, q_down):
        """脱励起係数 q_down(T) と詳細つり合いから得られる励起係数を衝突行列に加える"""
        boltzmann = numpy.exp(-(self.energy[upper] - self.energy[lower]) / (k_B * T_GRID))
        C_matrix[:, upper, lower] += q_down
        C_matrix[:, lower, upper] += q_down * self.g[upper] / self.g[lower] * boltzmann

    def cooling_per_ion(self, n_e, n_HI):
        """
        温度グリッド上の 1 イオンあたりの輝線冷却率 [erg s^-1]
        統計平衡 Σ_j n_j R_ji = n_i Σ_j R_ij（R = A + n_e C_e + n_HI C_H）を Σ n_i = 1 のもとで解く。
        """
        R = self.A[None, :, :] + n_e * self.C_e + n_HI * self.C_H
        R[:, 1:, 0] += 1e-30  # どの遷移ともつながらない準位があっても行列が特異にならないようにする
        L = numpy.swapaxes(R, 1, 2).copy()
        diagonal = numpy.arange(self.N_levels)
        L[:, diagonal, diagonal] -= numpy.sum(R, axis=2)
        L[:, 0, :] = 1.0  # 規格化条件 Σ n_i = 1 で 1 本目の式を置き換える
        b = numpy.zeros((len(T_GRID), self.N_levels, 1))
        b[:, 0, 0] = 1.0
        populations = numpy.linalg.solve(L, b)[:, :, 0]
        return numpy.maximum(populations, 0.0) @ self.radiated_power


CACHE_RELATIVE_TOLERANCE = 1e-3  # n_e, n_HI の相対変化がこれ未満なら直前の表を再利用する
MINIMUM_ION_FRACTION = 1e-8      # 割合がこれ未満のイオンは冷却に寄与しないとみなし、準位占有率を解かない
LOG_COOLING_FLOOR = -300.0       # 冷却率ゼロ（データなし）を表す log10 Λ


def _close(x, x_cached):
    return abs(x - x_cached) <= CACHE_RELATIVE_TOLERANCE * max(abs(x), abs(x_cached), 1e-30)


class LineCoolingSet:
    """
    複数イオンの輝線冷却をまとめて扱う。
    names: CHIANTI のイオン名のリスト。データがないイオンは None として冷却ゼロとする。
    """
    def __init__(self, names, N_levels_max=N_LEVELS_MAX):
        self.names = list(names)
        self.ions = []
        self.missing = []
        for name in self.names:
            if line_cooling_data_available(name):
                try:
                    self.ions.append(LineCoolingIon(name, N_levels_max))
                    continue
                except Exception as error:  # 想定外の形式のファイルは冷却ゼロとして進める
                    print(f"line_cooling: failed to load {name} ({error}); the ion is ignored", flush=True)
            else:
                self.missing.append(name)
            self.ions.append(None)
        # イオンごとに直前に作った表のキャッシュ: [n_e, n_HI, log10 Λ(T_GRID)]
        self._cache = [[None, None, None] for _ in self.ions]

    def table(self, n_e, n_HI, fractions):
        """
        電子密度 n_e と H I 数密度 n_HI [cm^-3] における各イオンの輝線冷却率の表
        fractions: 各イオンの元素内での割合（長さ len(names)）。MINIMUM_ION_FRACTION 未満のイオンは解かない。
        戻り値: 形状 (len(names), len(T_GRID)) の log10 Λ [erg s^-1]（イオン 1 個あたり）
        局所反復の中で n_e, n_HI はほとんど変わらないため、イオンごとに直前の表をキャッシュして再利用する。
        """
        table = numpy.full((len(self.ions), len(T_GRID)), LOG_COOLING_FLOOR)
        for k, ion in enumerate(self.ions):
            if ion is None or fractions[k] < MINIMUM_ION_FRACTION:
                continue
            cache = self._cache[k]
            if cache[2] is None or not (_close(n_e, cache[0]) and _close(n_HI, cache[1])):
                cache[:] = [n_e, n_HI, numpy.log10(numpy.maximum(ion.cooling_per_ion(n_e, n_HI), 1e-300))]
            table[k] = cache[2]
        return table

    @staticmethod
    def rate(T_gas, table):
        """table を log T について線形内挿し、温度 T_gas での Λ [erg s^-1]（イオン 1 個あたり、長さ len(names)）を返す"""
        log_T = numpy.clip(numpy.log10(T_gas), LOG_T_GRID[0], LOG_T_GRID[-1])
        k = min(numpy.searchsorted(LOG_T_GRID, log_T), len(LOG_T_GRID) - 1)
        k = max(k, 1)
        w = (log_T - LOG_T_GRID[k - 1]) / (LOG_T_GRID[k] - LOG_T_GRID[k - 1])
        return 10**((1.0 - w) * table[:, k - 1] + w * table[:, k])
