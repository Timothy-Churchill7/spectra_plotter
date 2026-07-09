#!/usr/bin/env python3
"""
spectra_core.py

Core parsing + plotting for the Plotter web app. Handles the three
"upload-and-plot" data types:

    * uvvis    — UV-Vis absorption spectra (xy, may overlay multiple traces)
    * emission — emission / PL spectra     (xy, may overlay multiple traces)
    * lifetime — TCSPC luminescence decay  (single trace + bi/tri-exp fit)

All traces are peak-normalized (peak = 1.0). The lifetime fit keeps whichever
of a bi- or tri-exponential model fits better (by BIC) and reports R^2 plus the
amplitude/lifetime coefficients.

This module is imported by app.py. It reuses the same tolerant file loader the
original spectra_plotter.py used so instrument-export metadata headers are
handled automatically.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
# Use the object-oriented Figure API (not pyplot) so plotting is thread-safe
# under Flask's threaded dev server — pyplot's global state is not.
from matplotlib.figure import Figure

# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
# Color cycle used when several traces are overlaid on one plot.
TRACE_COLORS = ["#2E86C1", "#E67E22", "#27AE60", "#8E44AD",
                "#C0392B", "#16A085", "#D4AC0D", "#5D6D7E"]
LINE_WIDTH = 1.4
TEXT_COLOR = "#333333"

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10.5,
    "text.color": TEXT_COLOR,
    "axes.edgecolor": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.linewidth": 0.7,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

# Default labels per type. These are also written (as editable constants) into
# the generated Colab scripts, so users can change titles/labels/sizes there.
LABELS = {
    "uvvis": {
        "xlabel": "Wavelength (nm)",
        "ylabel": "Normalized Absorbance",
        "title": "UV-Vis Absorption Spectrum",
        "logy": False,
    },
    "emission": {
        "xlabel": "Wavelength (nm)",
        "ylabel": "Normalized Emission Intensity (a.u.)",
        "title": "Emission Spectrum",
        "logy": False,
    },
    "lifetime": {
        "xlabel": "Time (ns)",
        "ylabel": "Normalized Intensity (a.u.)",
        "title": "Luminescence Decay",
        "logy": True,
    },
}


# ----------------------------------------------------------------------
# File loading — tolerant of instrument-export metadata headers
# ----------------------------------------------------------------------
def _sniff_delimiter(line: str):
    for delim in ("\t", ";", ","):
        if delim in line:
            return delim
    return None


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
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        tokens = [t for t in _split(stripped, delim) if t != ""]
        if len(tokens) >= 2 and _looks_numeric(tokens[0]) and _looks_numeric(tokens[1]):
            return i
    return None


def load_table(path: Path) -> pd.DataFrame:
    """Load a csv/txt/xlsx file into a numeric DataFrame, skipping any
    instrument metadata preamble and picking up a header row if present."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        raw = pd.read_excel(path, header=None)
        rows = raw.astype(str).values.tolist()
        data_start = None
        for i, row in enumerate(rows):
            tokens = [t for t in row if t not in ("nan", "")]
            if len(tokens) >= 2 and _looks_numeric(tokens[0]) and _looks_numeric(tokens[1]):
                data_start = i
                break
        if data_start is None:
            raise ValueError(f"Could not find numeric data in {path.name}")
        header = None
        if data_start > 0:
            header = [str(x) for x in raw.iloc[data_start - 1].tolist()]
        df = raw.iloc[data_start:].reset_index(drop=True)
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
        if header:
            df.columns = header[: df.shape[1]]
        return df

    lines = _read_raw_lines(path)
    delim = _find_delimiter(lines)
    data_start = _find_data_start(lines, delim)
    if data_start is None:
        raise ValueError(f"Could not find numeric data rows in {path.name}")

    header = None
    if data_start > 0:
        header = [t for t in _split(lines[data_start - 1], delim) if t != ""]

    sep = delim if delim else r"\s+"
    df = pd.read_csv(path, sep=sep, engine="python", skiprows=data_start, header=None)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if header:
        df.columns = header[: df.shape[1]]
    return df


# ----------------------------------------------------------------------
# Column selection
# ----------------------------------------------------------------------
def find_baseline_column(df: pd.DataFrame):
    for i, c in enumerate(df.columns):
        if i == 0:
            continue
        if re.search(r"baseline|\bblank\b", str(c), re.IGNORECASE):
            return i
    return None


def select_signal_column(df: pd.DataFrame, baseline_idx=None) -> int:
    candidates = [i for i in range(1, df.shape[1]) if i != baseline_idx]
    return candidates[0] if candidates else 1


def extract_xy(df: pd.DataFrame, data_type: str):
    """Return (x, y, note) for a single file. For uvvis, subtracts a baseline
    column if one is present."""
    if df.shape[1] < 2:
        raise ValueError("Expected at least 2 numeric columns.")
    baseline_idx = find_baseline_column(df)
    y_idx = select_signal_column(df, baseline_idx)

    x = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, y_idx].to_numpy(dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]

    note = None
    if data_type == "uvvis" and baseline_idx is not None:
        baseline = df.iloc[:, baseline_idx].to_numpy(dtype=float)[order]
        y = y - baseline
        note = "baseline-subtracted"
    return x, y, note


def detect_time_unit(df: pd.DataFrame) -> str:
    header = str(df.columns[0]).lower()
    for unit in ("ns", "µs", "us", "ms", "ps", "s"):
        if unit in header:
            return unit
    return "ns"


# ----------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------
def normalize(y: np.ndarray) -> np.ndarray:
    peak = np.nanmax(np.abs(y))
    if peak == 0:
        return y
    return y / peak


# ----------------------------------------------------------------------
# Lifetime decay: trim dead time, fit bi/tri-exponential decay + R^2
# ----------------------------------------------------------------------
def trim_leading_baseline(x, y, rise_fraction=0.03, window=5):
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
    return (a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)
            + a3 * np.exp(-t / tau3) + c)


def _r_squared(y, y_fit):
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def fit_lifetime(x, y, unit="ns"):
    """Fit the decay tail with bi- and tri-exponential models, keep whichever
    is meaningfully better by BIC. Returns a dict with the fitted curve, R^2,
    and human-readable coefficient annotation, or None if neither converges."""
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
            popt, _ = curve_fit(func, t_tail, y_tail, p0=p0,
                                bounds=(lo, hi), maxfev=40000)
        except Exception:
            return None
        y_fit = func(t_tail, *popt)
        resid = y_tail - y_fit
        sse = float(np.sum(resid ** 2))
        n, k = len(y_tail), len(p0)
        if sse <= 0:
            return None
        bic = n * np.log(sse / n) + k * np.log(n)
        return {"popt": popt, "y_fit": y_fit, "bic": bic,
                "n_components": n_components, "r2": _r_squared(y_tail, y_fit)}

    p0_bi = [0.7 * peak, span * 0.1, 0.3 * peak, span * 0.4, tail_floor]
    p0_tri = [0.5 * peak, span * 0.05, 0.3 * peak, span * 0.2,
              0.2 * peak, span * 0.6, tail_floor]

    bi = try_fit(_biexp, p0_bi, 2)
    tri = try_fit(_triexp, p0_tri, 3)

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
    baseline = float(popt[-1])

    order = np.argsort(taus)
    amps, taus = amps[order], taus[order]
    total = amps.sum()
    weights = amps / total if total > 0 else amps
    tau_avg = float(np.sum(amps * taus) / total) if total > 0 else float(np.mean(taus))

    kind = "Tri-exponential" if n == 3 else "Bi-exponential"
    lines = [f"{kind} fit   R² = {chosen['r2']:.4f}"]
    for i, (a, t) in enumerate(zip(amps, taus), start=1):
        lines.append(f"a{i} = {a:.3f},  τ{i} = {t:.2f} {unit}  ({weights[i-1]*100:.0f}%)")
    lines.append(f"c = {baseline:.3f}")
    lines.append(f"avg τ = {tau_avg:.2f} {unit}")

    return {
        "x_fit": x[peak_idx:],
        "y_fit": chosen["y_fit"],
        "kind": kind,
        "r2": chosen["r2"],
        "amps": [float(a) for a in amps],
        "taus": [float(t) for t in taus],
        "baseline": baseline,
        "tau_avg": tau_avg,
        "annotation": "\n".join(lines),
    }


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_spectra(traces, data_type, output_path, unit=None, note=None):
    """traces: list of (x, y, label). Renders a normalized plot to output_path.
    Returns fit_info (lifetime only) or None."""
    labels = dict(LABELS[data_type])
    if data_type == "lifetime" and unit:
        labels["xlabel"] = f"Time ({unit})"

    fig = Figure(figsize=(6.0, 4.3), dpi=200)
    ax = fig.subplots()
    fit_info = None
    show_legend = len(traces) > 1

    if data_type == "lifetime":
        # Single decay trace + fit.
        x, y, label = traces[0]
        x, y = trim_leading_baseline(x, y)
        y = normalize(y)
        fit_info = fit_lifetime(x, y, unit=unit or "ns")

        floor = max(np.nanmin(y[y > 0]) if np.any(y > 0) else 1e-4, 1e-6)
        y_plot = np.clip(y, floor, None)
        ax.semilogy(x, y_plot, color=TRACE_COLORS[0], linewidth=LINE_WIDTH,
                    solid_capstyle="round", label="Data")
        ax.set_ylim(bottom=floor * 0.8, top=1.3)

        if fit_info:
            ax.semilogy(fit_info["x_fit"], np.clip(fit_info["y_fit"], floor, None),
                        color=TEXT_COLOR, linewidth=1.1, linestyle="--",
                        dashes=(4, 2), label=f"{fit_info['kind']} fit")
            ax.legend(loc="upper right", fontsize=9, handlelength=1.6, labelspacing=0.4)
            ax.text(0.04, 0.04, fit_info["annotation"], transform=ax.transAxes,
                    fontsize=8, ha="left", va="bottom", color=TEXT_COLOR, linespacing=1.7)
    else:
        # uvvis / emission — one or more normalized traces.
        allmins = []
        for i, (x, y, label) in enumerate(traces):
            yn = normalize(y)
            allmins.append(np.nanmin(yn))
            ax.plot(x, yn, color=TRACE_COLORS[i % len(TRACE_COLORS)],
                    linewidth=LINE_WIDTH, solid_capstyle="round", label=label)
        lo = min(allmins) if allmins else 0.0
        pad = 0.05
        ax.set_ylim(lo - pad, 1.0 + pad)
        if show_legend:
            ax.legend(loc="best", fontsize=9, handlelength=1.6, labelspacing=0.4)

    if note:
        ax.text(0.97, 0.95, note, transform=ax.transAxes, fontsize=7.5,
                ha="right", va="top", color="#888888", style="italic")

    ax.set_xlabel(labels["xlabel"], fontsize=14, labelpad=6)
    ax.set_ylabel(labels["ylabel"], fontsize=14, labelpad=6)
    ax.set_title(labels["title"], fontsize=16, pad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fit_info


# ----------------------------------------------------------------------
# High-level entry used by the web app
# ----------------------------------------------------------------------
def process_spectra(input_paths, data_type, output_path, trace_labels=None):
    """Load one or more files, build normalized traces, render a plot.

    trace_labels: optional list of legend names aligned with input_paths
    (defaults to each file's stem).

    Returns dict: {"data_type", "unit", "note", "fit_info", "labels",
                   "trace_labels"}.
    """
    if isinstance(input_paths, (str, Path)):
        input_paths = [input_paths]

    unit = None
    note = None
    traces = []
    for i, p in enumerate(input_paths):
        p = Path(p)
        df = load_table(p)
        x, y, n = extract_xy(df, data_type)
        if n:
            note = n
        if data_type == "lifetime":
            unit = detect_time_unit(df)
        if trace_labels and i < len(trace_labels) and trace_labels[i]:
            label = trace_labels[i]
        else:
            label = p.stem
        traces.append((x, y, label))

    if data_type == "lifetime" and len(traces) > 1:
        # A tri-exponential fit only makes sense on a single decay curve.
        traces = traces[:1]

    fit_info = plot_spectra(traces, data_type, output_path, unit=unit, note=note)
    return {
        "data_type": data_type,
        "unit": unit,
        "note": note,
        "fit_info": fit_info,
        "labels": LABELS[data_type],
        "trace_labels": [t[2] for t in traces],
    }
