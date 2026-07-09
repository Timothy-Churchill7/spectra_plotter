#!/usr/bin/env python3
"""
colab.py

Generates self-contained, Google-Colab-ready Python scripts that reproduce each
plot the web app makes. Every script:

  * has an obvious `INSERT DATA PATH HERE` blank at the top,
  * exposes editable CONFIG constants for every title, label, font, size and
    color (mirroring the in-app styling tool),
  * needs only the libraries Colab already ships with (pandas, numpy, scipy,
    matplotlib) plus openpyxl for .xlsx.

The scripts are intentionally standalone (they do NOT import this project) so a
user can paste them straight into a Colab cell.
"""

# A compact, tolerant loader shared by the spectra scripts. Kept as plain text
# so it can be embedded verbatim into the generated files.
_LOADER = r'''
import re
from pathlib import Path
import numpy as np
import pandas as pd


def _looks_numeric(tok):
    try:
        float(str(tok).replace(",", "."))
        return True
    except (ValueError, AttributeError):
        return False


def load_table(path):
    """Load csv/txt/xlsx into a numeric DataFrame, skipping instrument
    metadata preambles and picking up a header row if present."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        raw = pd.read_excel(path, header=None)
        rows = raw.astype(str).values.tolist()
        start = None
        for i, row in enumerate(rows):
            toks = [t for t in row if t not in ("nan", "")]
            if len(toks) >= 2 and _looks_numeric(toks[0]) and _looks_numeric(toks[1]):
                start = i
                break
        if start is None:
            raise ValueError("No numeric data found in " + path.name)
        header = [str(x) for x in raw.iloc[start - 1].tolist()] if start > 0 else None
        df = raw.iloc[start:].reset_index(drop=True).apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
        if header:
            df.columns = header[: df.shape[1]]
        return df

    with open(path, "r", errors="ignore") as fh:
        lines = fh.readlines()
    delim = None
    for line in lines[:30]:
        for d in ("\t", ";", ","):
            if d in line:
                delim = d
                break
        if delim:
            break

    def split(line):
        return re.split(r"\s+", line.strip()) if delim is None else line.strip().split(delim)

    start = None
    for i, line in enumerate(lines):
        toks = [t for t in split(line) if t != ""]
        if len(toks) >= 2 and _looks_numeric(toks[0]) and _looks_numeric(toks[1]):
            start = i
            break
    if start is None:
        raise ValueError("No numeric data rows found in " + path.name)
    header = [t for t in split(lines[start - 1]) if t != ""] if start > 0 else None
    df = pd.read_csv(path, sep=(delim if delim else r"\s+"), engine="python",
                     skiprows=start, header=None).apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if header:
        df.columns = header[: df.shape[1]]
    return df


def normalize(y):
    peak = np.nanmax(np.abs(y))
    return y if peak == 0 else y / peak
'''


# ----------------------------------------------------------------------
# Shared style CONFIG block (mirrors the in-app styling tool)
# ----------------------------------------------------------------------
def _style_config(style, extra=""):
    return (
        "# ----- Titles / labels (None = auto) -----\n"
        f"TITLE   = {style.get('title')!r}\n"
        f"X_LABEL = {style.get('xlabel')!r}\n"
        f"Y_LABEL = {style.get('ylabel')!r}\n\n"
        "# ----- Text, size & color (edit freely) -----\n"
        f"FONT_FAMILY = {style['font_family']!r}\n"
        f"TITLE_SIZE  = {style['title_size']}\n"
        f"LABEL_SIZE  = {style['label_size']}\n"
        f"TICK_SIZE   = {style['tick_size']}\n"
        f"LEGEND_SIZE = {style['legend_size']}\n"
        f"ANNOT_SIZE  = {style['annot_size']}\n"
        f"TEXT_COLOR  = {style['text_color']!r}\n"
        f"LINE_WIDTH  = {style['line_width']}\n"
        f"FIG_W, FIG_H = {style['fig_w']}, {style['fig_h']}\n"
        f"SHOW_GRID   = {bool(style['show_grid'])}\n"
        f"SHOW_LEGEND = {bool(style['show_legend'])}\n"
        f"LEGEND_LOC  = {style['legend_loc']!r}\n"
        + extra
    )


# ----------------------------------------------------------------------
# UV-Vis / Emission (xy, one or more overlaid & possibly mixed traces)
# ----------------------------------------------------------------------
_XY_HEADER = '''#!/usr/bin/env python3
"""
Normalized spectra plot — reproducible figure for Google Colab.

HOW TO USE
----------
1. Upload your data file(s) to Colab (folder icon on the left, or drag-drop).
2. Fill in each "INSERT DATA PATH HERE" in SPECTRA below with the file path.
   Each entry already carries its data "type" and legend "label" — add or
   remove entries to change what is overlaid. Mixing "uvvis" and "emission"
   is fine; the legend is annotated with the kind automatically.
3. Run the cell. Edit the CONFIG constants to restyle titles, labels, fonts,
   sizes and colors.

UV-Vis ("uvvis") traces are cropped to the UVVIS_MIN..UVVIS_MAX window and
normalized within it; emission traces are normalized over their full range.
"""

# ============================ CONFIG =================================
'''

_XY_BODY = '''
# ----- UV-Vis normalization window (nm) -----
UVVIS_MIN = __UVMIN__
UVVIS_MAX = __UVMAX__

SAVE_AS = "plot.png"     # set to None to skip saving
# =====================================================================

__LOADER__

import matplotlib.pyplot as plt

COLORS = ["#2E86C1", "#E67E22", "#27AE60", "#8E44AD",
          "#C0392B", "#16A085", "#D4AC0D", "#5D6D7E"]
KIND = {"uvvis": "Absorbance", "emission": "Emission"}
AUTO_TITLE = {"uvvis": "UV-Vis Absorption Spectrum", "emission": "Emission Spectrum"}
AUTO_YLABEL = {"uvvis": "Normalized Absorbance",
               "emission": "Normalized Emission Intensity (a.u.)"}

types = [s["type"] for s in SPECTRA]
unique = list(dict.fromkeys(types))
mixed = len(unique) > 1

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=150)
for i, s in enumerate(SPECTRA):
    df = load_table(s["path"])
    x = df.iloc[:, 0].to_numpy(float)
    y = df.iloc[:, 1].to_numpy(float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if s["type"] == "uvvis":
        m = (x >= UVVIS_MIN) & (x <= UVVIS_MAX)
        if m.sum() >= 2:
            x, y = x[m], y[m]
    y = normalize(y)
    color = s.get("color") or COLORS[i % len(COLORS)]
    label = s["label"] + ((" (%s)" % KIND[s["type"]]) if mixed else "")
    ax.plot(x, y, color=color, linewidth=LINE_WIDTH, solid_capstyle="round", label=label)

title = TITLE or ("Normalized Spectra" if mixed else AUTO_TITLE[unique[0]])
xlabel = X_LABEL or "Wavelength (nm)"
ylabel = Y_LABEL or ("Normalized Signal (a.u.)" if mixed else AUTO_YLABEL[unique[0]])

ax.set_title(title, fontsize=TITLE_SIZE, fontfamily=FONT_FAMILY, color=TEXT_COLOR, pad=10)
ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontfamily=FONT_FAMILY, color=TEXT_COLOR, labelpad=6)
ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontfamily=FONT_FAMILY, color=TEXT_COLOR, labelpad=6)
ax.tick_params(labelsize=TICK_SIZE, colors=TEXT_COLOR)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontfamily(FONT_FAMILY)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(TEXT_COLOR)
if SHOW_GRID:
    ax.grid(True, alpha=0.3, linewidth=0.5)
if SHOW_LEGEND and ax.get_legend_handles_labels()[0]:
    leg = ax.legend(loc=LEGEND_LOC, fontsize=LEGEND_SIZE, frameon=False)
    for t in leg.get_texts():
        t.set_fontfamily(FONT_FAMILY)
        t.set_color(TEXT_COLOR)
ax.margins(x=0.02)
fig.tight_layout()
if SAVE_AS:
    fig.savefig(SAVE_AS, dpi=300, bbox_inches="tight")
plt.show()
'''


# ----------------------------------------------------------------------
# Lifetime (single decay + bi/tri-exponential fit, R^2)
# ----------------------------------------------------------------------
_LIFETIME_HEADER = '''#!/usr/bin/env python3
"""
Luminescence lifetime plot — reproducible figure for Google Colab.

HOW TO USE
----------
1. Upload your decay file to Colab and set DATA_PATH below.
2. Run the cell. The tail after the peak is fit with a bi- and a
   tri-exponential model; whichever fits better (by BIC) is kept, and the
   R^2, amplitudes (a_i), lifetimes (tau_i) and average tau are printed on the
   plot. Set FORCE_COMPONENTS = 3 to force a triexponential fit.
3. Edit the CONFIG constants to restyle titles, labels, fonts, sizes, colors.
"""

# ============================ CONFIG =================================
DATA_PATH = "INSERT DATA PATH HERE"
FORCE_COMPONENTS = None      # None = auto-pick 2 vs 3; or set 2 or 3
'''

_LIFETIME_BODY = '''
DATA_COLOR = __DATACOLOR__
UNIT = __UNIT__

SAVE_AS = "lifetime.png"     # set to None to skip saving
# =====================================================================

__LOADER__

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def biexp(t, a1, tau1, a2, tau2, c):
    return a1*np.exp(-t/tau1) + a2*np.exp(-t/tau2) + c


def triexp(t, a1, tau1, a2, tau2, a3, tau3, c):
    return a1*np.exp(-t/tau1) + a2*np.exp(-t/tau2) + a3*np.exp(-t/tau3) + c


def r_squared(y, yf):
    ss_res = np.sum((y - yf)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 0.0 if ss_tot <= 0 else 1.0 - ss_res/ss_tot


df = load_table(DATA_PATH)
x = df.iloc[:, 0].to_numpy(float)
y = df.iloc[:, 1].to_numpy(float)
order = np.argsort(x)
x, y = x[order], y[order]

peak_idx = int(np.argmax(y))
thresh = 0.03 * np.nanmax(y)
rise = 0
for i in range(peak_idx + 1):
    if y[i:i+5].mean() > thresh:
        rise = i
        break
x, y = x[max(rise-2, 0):], y[max(rise-2, 0):]
y = normalize(y)

peak_idx = int(np.argmax(y))
t = x[peak_idx:] - x[peak_idx]
yt = y[peak_idx:]
span = max(t[-1], 1e-6)
peak = yt[0] if yt[0] > 0 else np.nanmax(yt)
floor0 = float(np.mean(yt[-max(len(yt)//10, 5):]))
lo2, hi2 = [0, 1e-3], [5*peak, 10*span]


def do_fit(func, p0, n):
    lo = lo2 * n + [-1]
    hi = hi2 * n + [1]
    popt, _ = curve_fit(func, t, yt, p0=p0, bounds=(lo, hi), maxfev=40000)
    yf = func(t, *popt)
    sse = float(np.sum((yt - yf)**2))
    bic = len(yt)*np.log(sse/len(yt)) + len(p0)*np.log(len(yt))
    return dict(popt=popt, yf=yf, bic=bic, n=n, r2=r_squared(yt, yf))


p0_bi = [0.7*peak, span*0.1, 0.3*peak, span*0.4, floor0]
p0_tri = [0.5*peak, span*0.05, 0.3*peak, span*0.2, 0.2*peak, span*0.6, floor0]

fits = {}
try:
    fits[2] = do_fit(biexp, p0_bi, 2)
except Exception:
    pass
try:
    fits[3] = do_fit(triexp, p0_tri, 3)
except Exception:
    pass

if FORCE_COMPONENTS in (2, 3) and FORCE_COMPONENTS in fits:
    fit = fits[FORCE_COMPONENTS]
elif 2 in fits and 3 in fits:
    fit = fits[3] if (fits[2]["bic"] - fits[3]["bic"]) > 10 else fits[2]
else:
    fit = fits.get(3) or fits.get(2)

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=150)
floor = max(np.nanmin(y[y > 0]) if np.any(y > 0) else 1e-4, 1e-6)
ax.semilogy(x, np.clip(y, floor, None), color=DATA_COLOR, linewidth=LINE_WIDTH, label="Data")

if fit is not None:
    n = fit["n"]
    popt = fit["popt"]
    amps = popt[0:2*n:2]
    taus = popt[1:2*n:2]
    c = float(popt[-1])
    idx = np.argsort(taus)
    amps, taus = amps[idx], taus[idx]
    tot = amps.sum()
    w = amps/tot if tot > 0 else amps
    tau_avg = float(np.sum(amps*taus)/tot) if tot > 0 else float(np.mean(taus))
    kind = "Tri-exponential" if n == 3 else "Bi-exponential"
    ax.semilogy(x[peak_idx:], np.clip(fit["yf"], floor, None), color=TEXT_COLOR,
                linewidth=1.1, linestyle="--", dashes=(4, 2), label=kind + " fit")
    lines = ["%s fit   R² = %.4f" % (kind, fit["r2"])]
    for i in range(n):
        lines.append("a%d = %.3f,  τ%d = %.2f %s  (%.0f%%)"
                     % (i+1, amps[i], i+1, taus[i], UNIT, w[i]*100))
    lines.append("c = %.3f" % c)
    lines.append("avg τ = %.2f %s" % (tau_avg, UNIT))
    ax.text(0.04, 0.04, "\\n".join(lines), transform=ax.transAxes, fontsize=ANNOT_SIZE,
            ha="left", va="bottom", color=TEXT_COLOR, linespacing=1.7, fontfamily=FONT_FAMILY)
    print("\\n".join(lines))

ax.set_ylim(bottom=floor*0.8, top=1.3)
title = TITLE or "Luminescence Decay"
xlabel = X_LABEL or ("Time (%s)" % UNIT)
ylabel = Y_LABEL or "Normalized Intensity (a.u.)"
ax.set_title(title, fontsize=TITLE_SIZE, fontfamily=FONT_FAMILY, color=TEXT_COLOR, pad=10)
ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontfamily=FONT_FAMILY, color=TEXT_COLOR, labelpad=6)
ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontfamily=FONT_FAMILY, color=TEXT_COLOR, labelpad=6)
ax.tick_params(labelsize=TICK_SIZE, colors=TEXT_COLOR)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontfamily(FONT_FAMILY)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(TEXT_COLOR)
if SHOW_GRID:
    ax.grid(True, alpha=0.3, linewidth=0.5)
if SHOW_LEGEND:
    leg = ax.legend(loc=LEGEND_LOC, fontsize=LEGEND_SIZE, frameon=False)
    for tx in leg.get_texts():
        tx.set_fontfamily(FONT_FAMILY)
        tx.set_color(TEXT_COLOR)
ax.margins(x=0.02)
fig.tight_layout()
if SAVE_AS:
    fig.savefig(SAVE_AS, dpi=300, bbox_inches="tight")
plt.show()
'''


# ----------------------------------------------------------------------
# TEM histogram (from the exported quantum_dots.csv)
# ----------------------------------------------------------------------
_TEM_TEMPLATE = '''#!/usr/bin/env python3
"""
TEM quantum-dot size histogram — reproducible plot for Google Colab.

HOW TO USE
----------
1. Download the CSV from the TEM Dot Analyzer, upload it to Colab, and set
   DATA_PATH below.
2. Run the cell. It plots a histogram of the measured dot sizes and prints the
   mean / standard deviation. Edit the CONFIG block to restyle.

The CSV has a `length_nm` column (dot diameter in nm). If your run was not
scale-calibrated, switch VALUE_COLUMN to "length_px".
"""

# ============================ CONFIG =================================
DATA_PATH = "INSERT DATA PATH HERE"

VALUE_COLUMN = "length_nm"     # or "length_px" if not calibrated
N_BINS = 20

TITLE   = "TEM Quantum Dot Size Distribution"
X_LABEL = "Diameter (nm)"
Y_LABEL = "Count"

TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE  = 11
STAT_SIZE  = 10

BAR_COLOR  = "#2E86C1"
FIG_W, FIG_H = 6.0, 4.3
SAVE_AS = "tem_histogram.png"  # set to None to skip saving
# =====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(DATA_PATH)
vals = pd.to_numeric(df[VALUE_COLUMN], errors="coerce").dropna().to_numpy()

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=150)
ax.hist(vals, bins=N_BINS, color=BAR_COLOR, edgecolor="white", linewidth=0.6)

mean, std = float(np.mean(vals)), float(np.std(vals))
ax.axvline(mean, color="#C0392B", linewidth=1.2, linestyle="--")
ax.text(0.97, 0.95, "n = %d\\nmean = %.2f\\nstd = %.2f" % (len(vals), mean, std),
        transform=ax.transAxes, ha="right", va="top", fontsize=STAT_SIZE)

ax.set_title(TITLE, fontsize=TITLE_SIZE, pad=10)
ax.set_xlabel(X_LABEL, fontsize=LABEL_SIZE, labelpad=6)
ax.set_ylabel(Y_LABEL, fontsize=LABEL_SIZE, labelpad=6)
ax.tick_params(labelsize=TICK_SIZE)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
if SAVE_AS:
    fig.savefig(SAVE_AS, dpi=300, bbox_inches="tight")
plt.show()
print("n=%d  mean=%.3f  std=%.3f" % (len(vals), mean, std))
'''


# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------
def _build_xy_script(traces, style):
    lines = ["SPECTRA = ["]
    for tr in traces:
        lines.append(
            '    {"path": "INSERT DATA PATH HERE", "type": %r, "label": %r, "color": %r},'
            % (tr["data_type"], tr["label"], tr.get("color"))
        )
    lines.append("]\n")
    spectra_block = "\n".join(lines)

    body = (_XY_BODY
            .replace("__UVMIN__", repr(style.get("uvvis_min", 300.0)))
            .replace("__UVMAX__", repr(style.get("uvvis_max", 700.0)))
            .replace("__LOADER__", _LOADER))
    return _XY_HEADER + spectra_block + _style_config(style) + body


def _build_lifetime_script(trace, style):
    unit = trace.get("unit") or "ns"
    color = trace.get("color") or "#2E86C1"
    body = (_LIFETIME_BODY
            .replace("__DATACOLOR__", repr(color))
            .replace("__UNIT__", repr(unit))
            .replace("__LOADER__", _LOADER))
    return _LIFETIME_HEADER + _style_config(style) + body


def make_session_colab(sess):
    """Return a Colab-ready script reproducing the current session's plot."""
    if sess["mode"] == "lifetime":
        return _build_lifetime_script(sess["traces"][0], sess["style"])
    return _build_xy_script(sess["traces"], sess["style"])


def make_tem_colab_script():
    return _TEM_TEMPLATE
