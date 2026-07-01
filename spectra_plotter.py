#!/usr/bin/env python3
"""
spectra_plotter.py

Reads a data file (.csv, .xlsx, or .txt) exported from a UV-Vis, emission
(fluorescence/PL), or lifetime (TCSPC decay) instrument, figures out which
kind of measurement it is, normalizes the signal, and renders a clean,
publication-style PNG plot with matplotlib.

Usage:
    python spectra_plotter.py sample.csv
    python spectra_plotter.py sample.xlsx -o out.png
    python spectra_plotter.py sample.txt --type lifetime
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Styling: thin light-blue lines, thin professional/scientific text
# ----------------------------------------------------------------------
LINE_COLOR = "#6FB7E0"      # thin light blue
LINE_WIDTH = 1.2
TEXT_COLOR = "#333333"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10.5,
    "text.color": TEXT_COLOR,
    "axes.edgecolor": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.linewidth": 0.7,
    "axes.titlesize": 16,
    "axes.titleweight": "normal",
    "axes.labelsize": 14,
    "axes.labelweight": "normal",
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


# ----------------------------------------------------------------------
# File loading — tolerant of instrument-export metadata headers
# ----------------------------------------------------------------------
def _sniff_delimiter(line: str):
    for delim in ("\t", ";", ","):
        if delim in line:
            return delim
    return None  # falls back to whitespace splitting


def _split(line: str, delim):
    if delim is None:
        return re.split(r"\s+", line.strip())
    return line.strip().split(delim)


def _looks_numeric(token: str) -> bool:
    try:
        float(token.replace(",", "."))
        return True
    except (ValueError, AttributeError):
        return False


def _read_raw_lines(path: Path):
    with open(path, "r", errors="ignore") as f:
        return f.readlines()


def _find_delimiter(lines):
    for line in lines[:30]:
        if line.strip():
            d = _sniff_delimiter(line)
            if d:
                return d
    return None


def _find_data_start(lines, delim):
    """Return the index of the first line that looks like a numeric data row."""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        tokens = [t for t in _split(stripped, delim) if t != ""]
        if len(tokens) >= 2 and _looks_numeric(tokens[0]) and _looks_numeric(tokens[1]):
            return i
    return None


def load_table(path: Path) -> pd.DataFrame:
    """Load a csv/txt/xlsx file into a DataFrame of numeric columns, skipping
    any instrument metadata preamble and picking up a header row if present."""
    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        raw = pd.read_excel(path, header=None)
        lines_as_tokens = raw.astype(str).values.tolist()
        data_start = None
        for i, row in enumerate(lines_as_tokens):
            tokens = [t for t in row if t not in ("nan", "")]
            if len(tokens) >= 2 and _looks_numeric(tokens[0]) and _looks_numeric(tokens[1]):
                data_start = i
                break
        if data_start is None:
            raise ValueError(f"Could not find numeric data in {path}")
        header = None
        if data_start > 0:
            header = [str(x) for x in raw.iloc[data_start - 1].tolist()]
        df = raw.iloc[data_start:].reset_index(drop=True)
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
        if header:
            header = header[: df.shape[1]]
            df.columns = header
        return df

    # csv / txt
    lines = _read_raw_lines(path)
    delim = _find_delimiter(lines)
    data_start = _find_data_start(lines, delim)
    if data_start is None:
        raise ValueError(f"Could not find numeric data rows in {path}")

    header = None
    if data_start > 0:
        header = [t for t in _split(lines[data_start - 1], delim) if t != ""]

    sep = delim if delim else r"\s+"
    df = pd.read_csv(
        path, sep=sep, engine="python", skiprows=data_start, header=None
    )
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if header:
        header = header[: df.shape[1]]
        df.columns = header
    return df


# ----------------------------------------------------------------------
# Instrument-type detection
# ----------------------------------------------------------------------
def detect_type(df: pd.DataFrame) -> str:
    """Classify the data as 'uvvis', 'emission', or 'lifetime' using column
    header keywords plus shape-based heuristics on the values themselves."""
    headers = " ".join(str(c).lower() for c in df.columns)

    score = {"uvvis": 0, "emission": 0, "lifetime": 0}

    if re.search(r"\babs(orbance)?\b|%\s*t\b|transmittance", headers):
        score["uvvis"] += 3
    if re.search(r"\btime\b|\bns\b|\bus\b|\bµs\b|\bms\b|decay|lifetime|channel|tcspc", headers):
        score["lifetime"] += 3
    if re.search(r"\b(emission|fluorescence|photoluminescence|\bpl\b|cps)\b", headers):
        score["emission"] += 2
    if re.search(r"\bintensity\b|\bcounts\b", headers):
        score["emission"] += 1
        score["lifetime"] += 1
    if re.search(r"wavelength|\bnm\b", headers):
        score["uvvis"] += 1
        score["emission"] += 1

    x = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float)

    # Absorbance is almost always within roughly [-0.5, 5]
    if np.nanmax(y) <= 5 and np.nanmin(y) >= -0.5:
        score["uvvis"] += 1
    # Photon-counting signals (emission/lifetime) are usually large counts
    if np.nanmax(y) > 50:
        score["emission"] += 1
        score["lifetime"] += 1

    # Monotonic-decay-after-peak is the signature of a lifetime trace
    peak_idx = int(np.argmax(y))
    tail = y[peak_idx:]
    if len(tail) > 5:
        diffs = np.diff(tail)
        frac_decreasing = np.mean(diffs <= 0)
        if frac_decreasing > 0.85:
            score["lifetime"] += 3

    return max(score, key=score.get)


def detect_time_unit(df: pd.DataFrame) -> str:
    header = str(df.columns[0]).lower()
    for unit in ("ns", "us", "µs", "ms", "s"):
        if unit in header:
            return unit
    return "ns"


# ----------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------
def normalize(y: np.ndarray) -> np.ndarray:
    """Scale so the peak (max absolute value) sits at 1.0, preserving baseline."""
    peak = np.nanmax(np.abs(y))
    if peak == 0:
        return y
    return y / peak


# ----------------------------------------------------------------------
# Lifetime decay: trim dead-time baseline, fit multi-exponential decay
# ----------------------------------------------------------------------
def trim_leading_baseline(x: np.ndarray, y: np.ndarray, rise_fraction: float = 0.03, window: int = 5):
    """Drop the flat, near-zero region before the excitation pulse arrives
    (e.g. 0-40 ns of ~0 counts), which isn't part of the decay and would
    otherwise distort a curve fit. Keeps everything from just before the
    signal actually starts rising toward its peak."""
    peak = np.nanmax(y)
    if peak <= 0:
        return x, y
    peak_idx = int(np.argmax(y))
    threshold = rise_fraction * peak

    rise_idx = 0
    for i in range(peak_idx + 1):
        w = y[i:min(i + window, peak_idx + 1)]
        if w.size and w.mean() > threshold:
            rise_idx = i
            break

    start_idx = max(rise_idx - 2, 0)
    return x[start_idx:], y[start_idx:]


def _biexp(t, a1, tau1, a2, tau2, c):
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + c


def _triexp(t, a1, tau1, a2, tau2, a3, tau3, c):
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + a3 * np.exp(-t / tau3) + c


def fit_lifetime(x: np.ndarray, y: np.ndarray):
    """Fit the decay tail (from the peak onward) with a bi-exponential model,
    and a tri-exponential model, keeping whichever is meaningfully better by
    BIC. Returns a dict describing the fit, or None if neither converges."""
    peak_idx = int(np.argmax(y))
    t_tail = x[peak_idx:] - x[peak_idx]
    y_tail = y[peak_idx:]
    if len(t_tail) < 10:
        return None

    span = max(t_tail[-1], 1e-6)
    peak = y_tail[0] if y_tail[0] > 0 else np.nanmax(y_tail)
    tail_floor = float(np.mean(y_tail[-max(len(y_tail) // 10, 5):]))

    bounds_lo = [0, 1e-3]
    bounds_hi = [5 * peak, 10 * span]

    def try_fit(func, p0, n_components):
        lo = bounds_lo * n_components + [-1]
        hi = bounds_hi * n_components + [1]
        try:
            popt, _ = curve_fit(func, t_tail, y_tail, p0=p0, bounds=(lo, hi), maxfev=40000)
        except Exception:
            return None
        y_fit = func(t_tail, *popt)
        resid = y_tail - y_fit
        sse = float(np.sum(resid ** 2))
        n = len(y_tail)
        k = len(p0)
        if sse <= 0:
            return None
        bic = n * np.log(sse / n) + k * np.log(n)
        return {"popt": popt, "y_fit": y_fit, "bic": bic, "n_components": n_components}

    p0_bi = [0.7 * peak, span * 0.1, 0.3 * peak, span * 0.4, tail_floor]
    p0_tri = [0.5 * peak, span * 0.05, 0.3 * peak, span * 0.2, 0.2 * peak, span * 0.6, tail_floor]

    bi = try_fit(_biexp, p0_bi, 2)
    tri = try_fit(_triexp, p0_tri, 3)

    # Only prefer the tri-exponential model if it's a meaningfully better fit
    # (BIC lower by more than 10) — otherwise the simpler model wins.
    chosen = None
    if bi and tri:
        chosen = tri if (bi["bic"] - tri["bic"]) > 10 else bi
    else:
        chosen = bi or tri
    if chosen is None:
        return None

    n = chosen["n_components"]
    popt = chosen["popt"]
    amps = popt[0:2 * n:2]
    taus = popt[1:2 * n:2]
    baseline = popt[-1]

    order = np.argsort(taus)
    amps, taus = amps[order], taus[order]
    weights = amps / amps.sum() if amps.sum() > 0 else amps
    tau_avg = float(np.sum(amps * taus) / amps.sum()) if amps.sum() > 0 else float(np.mean(taus))

    kind = "Tri-exponential" if n == 3 else "Bi-exponential"
    lines = [kind + " fit"]
    for i, (a, t) in enumerate(zip(weights, taus), start=1):
        lines.append(f"τ{i} = {t:.2f} ({a * 100:.0f}%)")
    lines.append(f"avg τ = {tau_avg:.2f}")

    return {
        "x_fit": x[peak_idx:],
        "y_fit": chosen["y_fit"],
        "kind": kind,
        "annotation": "\n".join(lines),
    }


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
LABELS = {
    "uvvis": {
        "xlabel": "Wavelength (nm)",
        "ylabel": "Normalized Absorbance",
        "title": "UV–Vis Absorption Spectrum",
        "logy": False,
    },
    "emission": {
        "xlabel": "Wavelength (nm)",
        "ylabel": "Normalized Emission Intensity (a.u.)",
        "title": "Emission Spectrum",
        "logy": False,
    },
    "lifetime": {
        "xlabel": "Time",
        "ylabel": "Normalized Intensity (a.u.)",
        "title": "Luminescence Decay",
        "logy": True,
    },
}


def plot_spectrum(x, y, data_type: str, source_name: str, output_path: Path, fit_info: dict = None):
    y_norm = normalize(y)
    labels = LABELS[data_type]

    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=300)

    data_label = "Data" if fit_info else None
    if labels["logy"]:
        floor = max(np.nanmin(y_norm[y_norm > 0]) if np.any(y_norm > 0) else 1e-4, 1e-6)
        y_plot = np.clip(y_norm, floor, None)
        ax.semilogy(x, y_plot, color=LINE_COLOR, linewidth=LINE_WIDTH, solid_capstyle="round", label=data_label)
        ax.set_ylim(bottom=floor * 0.8, top=1.3)
    else:
        ax.plot(x, y_norm, color=LINE_COLOR, linewidth=LINE_WIDTH, solid_capstyle="round", label=data_label)
        pad = 0.05 * (np.nanmax(y_norm) - np.nanmin(y_norm) or 1)
        ax.set_ylim(np.nanmin(y_norm) - pad, np.nanmax(y_norm) + pad)

    if fit_info:
        ax.plot(
            fit_info["x_fit"], fit_info["y_fit"],
            color=TEXT_COLOR, linewidth=1.0, linestyle="--", dashes=(4, 2),
            label=f"{fit_info['kind']} fit",
        )
        ax.legend(loc="upper right", fontsize=8.5, handlelength=1.6, labelspacing=0.4)
        ax.text(
            0.04, 0.04, fit_info["annotation"],
            transform=ax.transAxes, fontsize=8, ha="left", va="bottom",
            color=TEXT_COLOR, linespacing=1.7,
        )

    ax.set_xlabel(labels["xlabel"], fontsize=14, labelpad=6)
    ax.set_ylabel(labels["ylabel"], fontsize=14, labelpad=6)
    ax.set_title(labels["title"], fontsize=16, pad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)

    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def process_file(input_path: str, output_path: str = None, force_type: str = None) -> Path:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = load_table(path)
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 numeric columns, found {df.shape[1]} in {path}")

    data_type = force_type or detect_type(df)

    x = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]

    fit_info = None
    if data_type == "lifetime":
        unit = detect_time_unit(df)
        labels = dict(LABELS["lifetime"])
        labels["xlabel"] = f"Time ({unit})"
        LABELS["lifetime"] = labels

        x, y = trim_leading_baseline(x, y)
        y = normalize(y)
        fit_info = fit_lifetime(x, y)

    out = Path(output_path) if output_path else path.with_suffix(".png")
    plot_spectrum(x, y, data_type, path.name, out, fit_info=fit_info)
    return out, data_type


def main():
    parser = argparse.ArgumentParser(description="Auto-detect and plot UV-Vis, emission, or lifetime spectra.")
    parser.add_argument("input", help="Path to a .csv, .xlsx, or .txt data file")
    parser.add_argument("-o", "--output", help="Output PNG path (default: same name as input, .png)")
    parser.add_argument("--type", choices=["uvvis", "emission", "lifetime"], help="Override auto-detection")
    args = parser.parse_args()

    out, data_type = process_file(args.input, args.output, args.type)
    print(f"Detected type: {data_type}")
    print(f"Saved plot to: {out}")


if __name__ == "__main__":
    main()
