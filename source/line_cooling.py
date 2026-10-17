"""
イオンの衝突励起による輝線冷却 (CHIANTI の準位・遷移・衝突強度データ、全元素共通)

各イオンについて低い方から N 準位を考え（N はイオンごとの表 database/line_cooling/n_levels.tsv、なければ N_LEVELS_MAX）、
電子衝突（および C I, C II の微細構造準位では
H 原子衝突）と自発放射の統計平衡から準位占有率を解き、冷却率 Λ = Σ_i n_i Σ_j A_ij ΔE_ij を求める。
計算を速くするため、衝突係数は温度グリッド上で読み込み時に準備しておき (単精度)、
LineCoolingSet.table(n_e, n_HI, fractions) で温度グリッド上の冷却率の表を作り、
LineCoolingSet.rate(T, table) で内挿して使う。表は log n_e の節点 (LOG_NE_NODE_STEP 間隔) ごとに解いてキャッシュし、
節点の間は log Λ を log n_e について線形内挿する (H 原子衝突のあるイオンと脱出確率を使うときはその場で解く)。
CHIANTI に衝突強度 (.scups) がないイオン (例: Fe XVI 以下の一部、Cr, Mn の多く) は冷却に寄与しないとみなす。
"""
import astropy.constants
import astropy.units
import numpy
import os
import scipy.interpolate

import constants

# 物理定数 (cgs、astropy.constants の CODATA 値)
h = astropy.constants.h.cgs.value  # プランク定数 [erg s]
c = astropy.constants.c.cgs.value  # 光速度 [cm s^-1]
k_B = astropy.constants.k_B.cgs.value  # ボルツマン定数 [erg K^-1]
eV = astropy.units.eV.to(astropy.units.erg)  # 電子ボルト [erg]
Ry = (astropy.constants.Ryd * astropy.constants.h * astropy.constants.c).cgs.value  # リュードベリ [erg]


N_LEVELS_MAX = 20       # n_levels.tsv にないイオンで考慮する準位数（低い方から）
# イオンごとの準位数 (make_line_cooling_levels.py が作る。冷却率が全準位の値と 2% 以内で一致する最小の準位数、上限 300)
N_LEVELS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database", "line_cooling", "n_levels.tsv")
_n_levels_table = None


def default_number_of_levels(name):
    """イオン name (CHIANTI のイオン名) で考慮する準位数 (n_levels.tsv の値、なければ N_LEVELS_MAX)"""
    global _n_levels_table
    if _n_levels_table is None:
        _n_levels_table = {}
        if os.path.exists(N_LEVELS_FILE):
            with open(N_LEVELS_FILE) as f:
                header = f.readline().rstrip("\n").split("\t")
                for line in f:
                    row = dict(zip(header, line.rstrip("\n").split("\t")))
                    _n_levels_table[row["ion"]] = int(row["N_levels"])
    return _n_levels_table.get(name, N_LEVELS_MAX)

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
    戻り値: {準位番号: (統計的重み g, エネルギー [cm^-1], 準位の名前 "配置 2S+1 L J")}
    観測値がない準位 (観測エネルギー < 0) は理論値を使う。
    """
    levels = {}
    # CHIANTI の固定幅書式 (i7, a30, a5, i5, a5, f5.1, f15.3, f15.3)。C IV では "," 以降、Ne I では空白区切りで
    # 余分な列が続くので、空白で分割せず固定幅で読む。
    for line in _read_until_terminator(filepath):
        index = int(line[0:7])
        configuration = line[7:37].strip()
        multiplicity, L = line[42:47].strip(), line[47:52].strip()
        J = float(line[52:57])
        energy_observed, energy_theory = float(line[57:72]), float(line[72:87])
        energy = energy_observed if energy_observed >= 0.0 else energy_theory
        J_text = f"{J:g}" if J != int(J) else f"{int(J)}"
        term = f"{configuration} {multiplicity}{L}{J_text}"
        levels[index] = (2.0 * J + 1.0, energy, term)
    return levels


def _read_wgfa(filepath):
    """放射遷移ファイル (.wgfa) を読み込む。戻り値: [(準位1, 準位2, A [s^-1], gf), ...]"""
    transitions = []
    for line in _read_until_terminator(filepath):
        columns = line.split()
        transitions.append((int(columns[0]), int(columns[1]), float(columns[4]), float(columns[3])))
    return transitions


def escape_probability(tau):
    """
    線中心の光学的厚さ tau (片側) に対する静的な媒質の脱出確率 β (ドップラー線、XSTAR の pescl と同じ式で片側の値)
      τ < 1: (1 − e^{−2τ}) / (2τ)、τ ≥ 1: 1 / [τ √π (1.2 + 0.5 √(ln τ) / (1 + τ/10⁵))]
    """
    tau = numpy.asarray(tau, dtype=float)
    small = numpy.where(tau < 1e-5, 1.0, -numpy.expm1(-2.0 * numpy.minimum(tau, 1.0)) / (2.0 * numpy.maximum(tau, 1e-300)))
    big_tau = numpy.maximum(tau, 1.0)
    large = 1.0 / (big_tau * numpy.sqrt(numpy.pi) * (1.2 + 0.5 * numpy.sqrt(numpy.log(big_tau)) / (1.0 + big_tau / 1.0e5)))
    return numpy.where(tau < 1.0, small, large)


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
    return os.path.join(constants.CHIANTI_DIR, name.split("_")[0], name)


def line_cooling_data_available(name):
    """輝線冷却の計算に必要な .elvlc, .wgfa, .scups が揃っているか"""
    directory = chianti_ion_directory(name)
    return all(os.path.exists(os.path.join(directory, f"{name}.{ext}")) for ext in ("elvlc", "wgfa", "scups"))


class LineCoolingIon:
    """1 つのイオンの多準位原子モデル (name は CHIANTI のイオン名、例: "c_2", "fe_25")"""
    def __init__(self, name, N_levels_max=None, log_T_grid=None):
        """
        N_levels_max: 考慮する準位数の上限 (None ならイオンごとの表 default_number_of_levels)
        log_T_grid:   衝突係数と冷却率の表の温度グリッド log10 T (None なら LOG_T_GRID。準位数の調査用)
        """
        self.name = name
        self.log_T_grid = LOG_T_GRID if log_T_grid is None else numpy.asarray(log_T_grid, dtype=float)
        T_GRID = 10**self.log_T_grid
        self.T_grid = T_GRID
        directory = chianti_ion_directory(name)
        levels = _read_elvlc(os.path.join(directory, f"{name}.elvlc"))
        if N_levels_max is None:
            N_levels_max = default_number_of_levels(name)
        N = min(N_levels_max, len(levels))
        self.N_levels = N
        self.N_levels_all = len(levels)
        self.g = numpy.array([levels[i + 1][0] for i in range(N)])
        self.energy = numpy.array([levels[i + 1][1] for i in range(N)]) * h * c  # [erg]
        self.term = [levels[i + 1][2] for i in range(N)]                             # 準位の名前（出力用）
        T = T_GRID[:, None]

        # 自発放射: A[i, j] は準位 i → j の遷移確率（エネルギーの高い方から低い方へ）
        self.A = numpy.zeros((N, N))
        gf = numpy.zeros((N, N))
        for level1, level2, A, gf_value in _read_wgfa(os.path.join(directory, f"{name}.wgfa")):
            i, j = level1 - 1, level2 - 1
            if i >= N or j >= N or i == j or A <= 0.0:
                continue
            upper, lower = (i, j) if self.energy[i] > self.energy[j] else (j, i)
            self.A[upper, lower] += A
            gf[upper, lower] += max(gf_value, 0.0)
        # 輝線の一覧 (上準位, 下準位, A [s^-1], エネルギー [erg], 吸収の振動子強度 f_lu) — 放射率の出力と光学的厚さに使う
        upper_index, lower_index = numpy.nonzero(self.A)
        self.line_upper = upper_index
        self.line_lower = lower_index
        self.line_A = self.A[upper_index, lower_index]
        self.line_energy = self.energy[upper_index] - self.energy[lower_index]
        self.line_f = gf[upper_index, lower_index] / self.g[lower_index]
        self.line_wavelength_angstrom = h * c / numpy.maximum(self.line_energy, 1e-300) * 1.0e8
        self.delta_E = numpy.maximum(self.energy[:, None] - self.energy[None, :], 0.0)
        # 各準位から自発放射で放出されるエネルギー Σ_j A_ij ΔE_ij [erg s^-1]
        self.radiated_power = numpy.sum(self.A * self.delta_E, axis=1)

        # 電子衝突: C_e[t, i, j] は温度 T_GRID[t] で準位 i → j の係数 [cm^3 s^-1] (メモリを抑えるため単精度)
        self.C_e = numpy.zeros((len(T_GRID), N, N), dtype=numpy.float32)
        for level1, level2, delta_E_Ry, transition_type, C, x_knots, y_knots in _read_scups(os.path.join(directory, f"{name}.scups")):
            i, j = level1 - 1, level2 - 1
            if i >= N or j >= N or i == j:
                continue
            lower, upper = (i, j) if self.energy[i] < self.energy[j] else (j, i)
            upsilon = numpy.maximum(effective_collision_strength(T_GRID, delta_E_Ry, transition_type, C, x_knots, y_knots), 0.0)
            q_down = 8.629e-6 / (self.g[upper] * numpy.sqrt(T_GRID)) * upsilon
            self._add_collision_pair(self.C_e, lower, upper, q_down)

        # H 原子衝突: C_H[t, i, j] [cm^3 s^-1] (データのあるイオンだけ。ほかは None)
        self.C_H = None
        for (ion_name, upper, lower), rate in H_DEEXCITATION_RATE.items():
            if ion_name == name and upper <= N:
                if self.C_H is None:
                    self.C_H = numpy.zeros((len(T_GRID), N, N), dtype=numpy.float32)
                self._add_collision_pair(self.C_H, lower - 1, upper - 1, rate(T_GRID))

    def _add_collision_pair(self, C_matrix, lower, upper, q_down):
        """脱励起係数 q_down(T) と詳細つり合いから得られる励起係数を衝突行列に加える"""
        boltzmann = numpy.exp(-(self.energy[upper] - self.energy[lower]) / (k_B * self.T_grid))
        C_matrix[:, upper, lower] += q_down
        C_matrix[:, lower, upper] += q_down * self.g[upper] / self.g[lower] * boltzmann

    def effective_A(self, beta=None):
        """
        脱出確率 beta (輝線の一覧 line_upper/line_lower と同じ順の配列) をかけた自発放射の行列 A β。
        beta が None なら光学的に薄い (β = 1)。閉じ込められた光子はその場で再吸収されるとみなす (正味の放射は A β)。
        """
        if beta is None:
            return self.A
        A = numpy.zeros_like(self.A)
        A[self.line_upper, self.line_lower] = self.line_A * beta
        return A

    def populations_at(self, n_e, n_HI, T_gas, beta=None):
        """温度 T_gas [K] での準位占有率（Σ = 1）。衝突係数は温度グリッドから log T で線形内挿する。beta は脱出確率 (effective_A)。"""
        grid = self.log_T_grid
        log_T = numpy.clip(numpy.log10(T_gas), grid[0], grid[-1])
        k = min(max(int(numpy.searchsorted(grid, log_T)), 1), len(grid) - 1)
        w = (log_T - grid[k - 1]) / (grid[k] - grid[k - 1])
        C_e = (1.0 - w) * self.C_e[k - 1].astype(float) + w * self.C_e[k].astype(float)
        R = self.effective_A(beta) + n_e * C_e
        if self.C_H is not None:
            R += n_HI * ((1.0 - w) * self.C_H[k - 1].astype(float) + w * self.C_H[k].astype(float))
        R[1:, 0] += 1e-30
        L = R.T.copy()
        diagonal = numpy.arange(self.N_levels)
        L[diagonal, diagonal] -= numpy.sum(R, axis=1)
        L[0, :] = 1.0
        b = numpy.zeros(self.N_levels)
        b[0] = 1.0
        return numpy.maximum(numpy.linalg.solve(L, b), 0.0)

    def line_emission(self, n_e, n_HI, T_gas, beta=None):
        """
        温度 T_gas での各輝線の光子放射率 [photons s^-1]（イオン 1 個あたり。beta があれば脱出する分 n_u A β）
        戻り値: (上準位 (1 始まり), 下準位 (1 始まり), エネルギー [eV], 光子放射率) の配列の組
        """
        populations = self.populations_at(n_e, n_HI, T_gas, beta)
        photons = populations[self.line_upper] * self.line_A * (1.0 if beta is None else beta)
        return self.line_upper + 1, self.line_lower + 1, self.line_energy / eV, photons

    def cooling_per_ion(self, n_e, n_HI, beta=None):
        """
        温度グリッド上の 1 イオンあたりの輝線冷却率 [erg s^-1]
        統計平衡 Σ_j n_j R_ji = n_i Σ_j R_ij（R = A β + n_e C_e + n_HI C_H）を Σ n_i = 1 のもとで解く。
        beta は脱出確率 (effective_A)。冷却は脱出する放射 Σ n_u A β ΔE。
        """
        A = self.effective_A(beta)
        radiated_power = numpy.sum(A * self.delta_E, axis=1) if beta is not None else self.radiated_power
        N_T = len(self.T_grid)
        cooling = numpy.zeros(N_T)
        diagonal = numpy.arange(self.N_levels)
        chunk = max(1, int(2.0e7 // (self.N_levels**2)))   # 一度に解く温度の数 (倍精度の作業配列を ~160 MB 以下に抑える)
        for start in range(0, N_T, chunk):
            stop = min(start + chunk, N_T)
            R = A[None, :, :] + n_e * self.C_e[start:stop].astype(float)
            if self.C_H is not None and n_HI > 0.0:
                R += n_HI * self.C_H[start:stop].astype(float)
            R[:, 1:, 0] += 1e-30  # どの遷移ともつながらない準位があっても行列が特異にならないようにする
            L = numpy.swapaxes(R, 1, 2).copy()
            L[:, diagonal, diagonal] -= numpy.sum(R, axis=2)
            L[:, 0, :] = 1.0  # 規格化条件 Σ n_i = 1 で 1 本目の式を置き換える
            b = numpy.zeros((stop - start, self.N_levels, 1))
            b[:, 0, 0] = 1.0
            populations = numpy.linalg.solve(L, b)[:, :, 0]
            cooling[start:stop] = numpy.maximum(populations, 0.0) @ radiated_power
        return cooling


CACHE_RELATIVE_TOLERANCE = 1e-3  # n_e, n_HI の相対変化がこれ未満なら直前の表を再利用する
MINIMUM_ION_FRACTION = 1e-8      # 割合がこれ未満のイオンは冷却に寄与しないとみなし、準位占有率を解かない
LOG_COOLING_FLOOR = -300.0       # 冷却率ゼロ（データなし）を表す log10 Λ
LOG_NE_NODE_STEP = 0.05          # 冷却率の表をキャッシュする log10 n_e の節点の間隔 (節点の間は log Λ を線形内挿)


def _close(x, x_cached):
    return abs(x - x_cached) <= CACHE_RELATIVE_TOLERANCE * max(abs(x), abs(x_cached), 1e-30)


class LineCoolingSet:
    """
    複数イオンの輝線冷却をまとめて扱う。
    names: CHIANTI のイオン名のリスト。データがないイオンは None として冷却ゼロとする。
    """
    def __init__(self, names, N_levels_max=None):
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
        # イオンごとに直前に作った表のキャッシュ: [n_e, n_HI, log10 Λ(T_GRID), β] (H 原子衝突か脱出確率を使うとき)
        self._cache = [[None, None, None, None] for _ in self.ions]
        # イオンごとの log n_e の節点の表のキャッシュ: {節点の番号: log10 Λ(T_GRID)}
        self._node_cache = [{} for _ in self.ions]

    def _node_table(self, k, node):
        cache = self._node_cache[k]
        if node not in cache:
            n_e = 10**(node * LOG_NE_NODE_STEP)
            cache[node] = numpy.log10(numpy.maximum(self.ions[k].cooling_per_ion(n_e, 0.0), 1e-300))
        return cache[node]

    def table(self, n_e, n_HI, fractions, betas=None):
        """
        電子密度 n_e と H I 数密度 n_HI [cm^-3] における各イオンの輝線冷却率の表
        fractions: 各イオンの元素内での割合（長さ len(names)）。MINIMUM_ION_FRACTION 未満のイオンは解かない。
        戻り値: 形状 (len(names), len(T_GRID)) の log10 Λ [erg s^-1]（イオン 1 個あたり）
        局所反復の中で n_e, n_HI はほとんど変わらないため、イオンごとに直前の表をキャッシュして再利用する。
        betas: イオンごとの脱出確率の配列 (None の要素や betas = None は光学的に薄い)。
        """
        table = numpy.full((len(self.ions), len(T_GRID)), LOG_COOLING_FLOOR)
        for k, ion in enumerate(self.ions):
            if ion is None or fractions[k] < MINIMUM_ION_FRACTION:
                continue
            beta = None if betas is None else betas[k]
            if beta is None and ion.C_H is None and n_e > 0.0:
                # log n_e の節点で解いた表を内挿する (n_HI によらない)
                x = numpy.log10(n_e) / LOG_NE_NODE_STEP
                node = int(numpy.floor(x))
                w = x - node
                table[k] = (1.0 - w) * self._node_table(k, node) + w * self._node_table(k, node + 1)
                continue
            cache = self._cache[k]
            same_beta = (beta is None and cache[3] is None) or (beta is not None and cache[3] is not None and numpy.array_equal(beta, cache[3]))
            if cache[2] is None or not (_close(n_e, cache[0]) and _close(n_HI, cache[1]) and same_beta):
                cache[:] = [n_e, n_HI, numpy.log10(numpy.maximum(ion.cooling_per_ion(n_e, n_HI, beta), 1e-300)), beta]
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
