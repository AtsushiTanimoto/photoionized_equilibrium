# 周波数積分した制動放射の Gaunt 因子

van Hoof et al. (2015, MNRAS, 449, 2112, "Accurate determination of the free-free Gaunt factor - II. Relativistic
Gaunt factors") の熱平均・周波数積分した Gaunt 因子 ⟨g_ff⟩(γ²) の表 (相対論補正込み)。
https://data.nublado.org/gauntff/ の `gauntff_freqint.tar.gz` から Z = 1〜30 のファイルを改変せずに取ったもの (2026-10-16 取得)。
各ファイルの著作権表示 (Peter A.M. van Hoof、Royal Observatory of Belgium) はそのまま残している。

形式: 先頭 5 行が magic number (20141008)、格子点数 (161)、log₁₀ γ² の開始値 (−6)、刻み (0.1 dex)、原子番号 Z。
`#` の行はコメント。データは「log₁₀ γ²  ⟨g_ff⟩」の 161 行 (log₁₀ γ² = −6〜10)。
γ² = Z² Ry / kT (Z はイオンの電荷)。制動放射の冷却率は Λ = 1.426×10⁻²⁷ √T Z² ⟨g_ff⟩ n_e n_ion [erg cm⁻³ s⁻¹]。
