"""
Lexington 2000 ベンチマークの各モデルを Cloudy (~/software/cloudy/c25.00) で計算し、run_lexington.py と同じ量を
output/cloudy_<model>_results.tsv に保存する (run_lexington.py の表と図に "Cloudy C25" として載る)。
  uv run python run_cloudy.py --model HII40
入力は lexington_reference.MODELS と同じ (黒体、全光度、内径、一様密度、球対称、ベンチマークの元素だけ。他の元素は off)。
iterate を 1 回 (微細構造線の光学的厚さのため、Cloudy の標準の設定)、外側は T < 1000 K か電子の割合 < 10^-2 で止める
(PN75 は外径で止める)。ダスト、分子、宇宙線は入れない (Cloudy の既定)。

輝線は主出力の最後の反復の "Intrinsic line intensities" の表 (print lines column) から、run_lexington.LINES と同じ
化学種・波長範囲の線を足す (Cloudy の波長は 2000 Å 以上で空気中の値だが、範囲は数 Å の幅をとっているので含まれる)。
⟨T[NpNe]⟩、T_inner、R_out、⟨He⁺⟩/⟨H⁺⟩ は save overview のゾーンごとの値から本コードと同じ式で計算する。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source"))  # source/ のモジュールを使う
import argparse
import math
import re
import subprocess
import time

import numpy

import constants
import lexington_reference
import run_lexington
import visualization


CLOUDY_DIR = "/Users/tanimoto/software/cloudy/c25.00"
CLOUDY_EXE = os.path.join(CLOUDY_DIR, "source", "cloudy.exe")
CLOUDY_NAMES = {"H": "hydrogen", "He": "helium", "Li": "lithium", "Be": "beryllium", "B": "boron", "C": "carbon", "N": "nitrogen",
                "O": "oxygen", "F": "fluorine", "Ne": "neon", "Na": "sodium", "Mg": "magnesium", "Al": "aluminium", "Si": "silicon",
                "P": "phosphorus", "S": "sulphur", "Cl": "chlorine", "Ar": "argon", "K": "potassium", "Ca": "calcium",
                "Sc": "scandium", "Ti": "titanium", "V": "vanadium", "Cr": "chromium", "Mn": "manganese", "Fe": "iron",
                "Co": "cobalt", "Ni": "nickel", "Cu": "copper", "Zn": "zinc"}


def cloudy_directory(model_name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloudy", model_name)


def write_input(model_name):
    reference = lexington_reference.MODELS[model_name]
    name = model_name.lower()
    lines = [f"title Lexington 2000 {model_name}",
             f"blackbody {reference['T_BB']:.0f}",
             f"luminosity total {math.log10(reference['luminosity']):.6f}"]
    if reference["matter_bounded"]:
        lines.append(f"radius {math.log10(reference['R_in']):.6f} {math.log10(reference['r_max']):.6f}")
    else:
        lines.append(f"radius {math.log10(reference['R_in']):.6f}")
    lines += [f"hden {math.log10(reference['n_H']):.6f}", "sphere"]
    for symbol, abundance in reference["abundances"].items():
        if symbol != "H":
            lines.append(f"element {CLOUDY_NAMES[symbol]} abundance {math.log10(abundance):.6f}")
    for symbol in constants.ELEMENT_SYMBOLS:
        if symbol not in reference["abundances"]:
            lines.append(f"element {CLOUDY_NAMES[symbol]} off")
    lines += ["iterate"]
    if not reference["matter_bounded"]:
        lines += ["stop temperature 1000", "stop efrac -2"]
    lines += ["print line faint -6", "print lines column", "print lines sort wavelength",
              f"save overview last \"{name}.ovr\"", f"save cooling last \"{name}.col\"", f"save heating last \"{name}.het\"",
              f"save each cooling last \"{name}.cole\""]
    for symbol in reference["abundances"]:
        if symbol not in ("H", "He"):
            lines.append(f"save element {CLOUDY_NAMES[symbol]} last \"{name}.{symbol.lower()}\"")
    directory = cloudy_directory(model_name)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{name}.in"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return directory, name


def run(model_name):
    directory, name = write_input(model_name)
    environment = dict(os.environ, CLOUDY_DATA_PATH=os.path.join(CLOUDY_DIR, "data") + "/")
    start = time.time()
    subprocess.run([CLOUDY_EXE, "-r", name], cwd=directory, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
    print(f"Cloudy {model_name}: {(time.time() - start) / 60.0:.2f} min -> {directory}", flush=True)
    return directory, name


def parse_wavelength(text):
    """Cloudy の波長の表記 (4861.32A, 15.5509m (µm), 0 (連続光)) を Å にする"""
    match = re.fullmatch(r"([0-9.]+)([Amc]?)", text)
    if match is None:
        return None
    value, unit = float(match.group(1)), match.group(2)
    return {"A": value, "m": value * 1.0e4, "c": value * 1.0e8, "": value}[unit]


def read_lines(path):
    """主出力の最後の "Intrinsic line intensities" の表: [(化学種, 波長 [Å], log 光度, Hβ 比)]"""
    with open(path) as f:
        text = f.read().split("\n")
    starts = [k for k, line in enumerate(text) if line.strip() == "Intrinsic line intensities"]
    lines = []
    for line in text[starts[-1] + 1:]:
        if "Emergent line intensities" in line:
            break
        match = re.match(r"^(.{4})\s+(\S+)\s+([-0-9.]+)\s+([-0-9.eE+]+)\s*$", line)
        if match:
            wavelength = parse_wavelength(match.group(2))
            if wavelength:
                lines.append((match.group(1), wavelength, float(match.group(3)), float(match.group(4))))
    return lines


def cloudy_species(ion_symbol):
    """本コードのイオン記号 (O06 など) → Cloudy の化学種のラベル ("O  3" など)"""
    match = re.fullmatch(r"([A-Z][a-z]?)(\d\d)", ion_symbol)
    element, electrons = match.group(1), int(match.group(2))
    stage = constants.ATOMIC_NUMBER[element] - electrons + 1
    return f"{element:<2}{stage:>2}"


def results(model_name, directory, name):
    reference = lexington_reference.MODELS[model_name]
    abundances = reference["abundances"]
    table = read_lines(os.path.join(directory, f"{name}.out"))
    out = {}
    hbeta = [L for species, wavelength, L, _ in table if species == "H  1" and abs(wavelength - 4861.3) < 0.5]
    out["Hbeta"] = 10**hbeta[0] / reference["Hbeta_unit"]
    for quantity in reference["rows"]:
        if quantity not in run_lexington.LINES:
            continue
        ion_symbol, ranges = run_lexington.LINES[quantity]
        species = cloudy_species(ion_symbol)
        total = sum(I for s, wavelength, _, I in table if s == species and any(lo <= wavelength <= hi for lo, hi in ranges))
        out[quantity] = total
    # ゾーンごとの値: depth [cm], Te, hden, eden, HII, HeII の列
    with open(os.path.join(directory, f"{name}.ovr")) as f:
        header = f.readline().lstrip("#").strip().split("\t")
    data = numpy.loadtxt(os.path.join(directory, f"{name}.ovr"), comments="#")
    column = {key: data[:, k] for k, key in enumerate(header)}
    r = reference["R_in"] + column["depth"]
    edges = numpy.concatenate([[reference["R_in"]], 0.5 * (r[1:] + r[:-1]), [r[-1] + 0.5 * (r[-1] - r[-2])]])
    volume = 4.0 / 3.0 * numpy.pi * (edges[1:]**3 - edges[:-1]**3)
    n_e, n_p = column["eden"], column["hden"] * column["HII"]
    n_Heplus = column["hden"] * abundances["He"] * column["HeII"]
    out["T_inner"] = column["Te"][0]
    out["<T[NpNe]>"] = numpy.sum(n_e * n_p * column["Te"] * volume) / numpy.sum(n_e * n_p * volume)
    if not reference["matter_bounded"]:
        k = numpy.argmax(column["HII"] < 0.5) if numpy.any(column["HII"] < 0.5) else len(r) - 1
        out["R_out"] = r[k] / reference["R_out_unit"]
    out["<He+>/<H+>"] = abundances["H"] / abundances["He"] * numpy.sum(n_e * n_Heplus * volume) / numpy.sum(n_e * n_p * volume)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HII40", choices=list(lexington_reference.MODELS))
    parser.add_argument("--parse-only", action="store_true", help="Cloudy を走らせず、既存の出力から結果を作り直す")
    args = parser.parse_args()
    if args.parse_only:
        directory, name = cloudy_directory(args.model), args.model.lower()
    else:
        directory, name = run(args.model)
    out = results(args.model, directory, name)
    path = os.path.join(visualization.OUTPUT_DIR, f"cloudy_{args.model}_results.tsv")
    os.makedirs(visualization.OUTPUT_DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write("quantity\tcloudy\tmedian\n")
        for quantity, value in out.items():
            median = lexington_reference.median(args.model, quantity)
            f.write(f"{quantity}\t{value:.6e}\t{median if median is not None else 'nan'}\n")
    print(path)
    for quantity, value in out.items():
        median = lexington_reference.median(args.model, quantity)
        print(f"  {quantity:22s} {value:10.4g}  median {median if median is not None else float('nan'):9.4g}")
