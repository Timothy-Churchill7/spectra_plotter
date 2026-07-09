#!/usr/bin/env python3
"""
colab.py

Generates self-contained, Google-Colab-ready Python scripts that reproduce each
plot the web app makes. Every script:

  * has an obvious `INSERT DATA PATH HERE` blank at the top,
  * exposes editable CONFIG constants for every label, title and font size,
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


def _subst(template, **kw):
    out = template
    for k, v in kw.items():
        out = out.replace("__%s__" % k, str(v))
    return out


# ----------------------------------------------------------------------
# UV-Vis / Emission (xy, one or more overlaid traces, peak-normalized)
# ----------------------------------------------------------------------
_XY_TEMPLATE = '''#!/usr/bin/env python3
"""
__TITLE__ — reproducible plot for Google Colab.

HOW TO USE
----------
1. Upload your data file(s) to Colab (folder icon on the left, or drag-drop).
2. Put the path(s) in DATA_PATHS below, replacing INSERT DATA PATH HERE.
   Add more paths to overlay several spectra on the same axes.
3. Run the cell. Edit anything in the CONFIG block to restyle the figure.

Accepts .csv, .txt or .xlsx exported from the instrument. The first column is
X (wavelength), the second is the signal. Each trace is normalized so its peak
= 1.0.
"""

# ============================ CONFIG =================================
DATA_PATHS = [
    "INSERT DATA PATH HERE",
    # "second_sample.csv",   # <- uncomment / add more to overlay
]

# Optional: give each trace a legend name. Leave empty to use file names.
TRACE_LABELS = []            # e.g. ["Sample A", "Sample B"]

TITLE   = "__TITLE__"
X_LABEL = "__XLABEL__"
Y_LABEL = "__YLABEL__"

TITLE_SIZE  = 16            # change text sizes freely
LABEL_SIZE  = 14
TICK_SIZE   = 11
LEGEND_SIZE = 10

LINE_WIDTH  = 1.4
FIG_W, FIG_H = 6.0, 4.3
SAVE_AS = "plot.png"        # set to None to skip saving
# =====================================================================

__LOADER__

import matplotlib.pyplot as plt

COLORS = ["#2E86C1", "#E67E22", "#27AE60", "#8E44AD",
          "#C0392B", "#16A085", "#D4AC0D", "#5D6D7E"]

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=150)

for i, path in enumerate(DATA_PATHS):
    df = load_table(path)
    x = df.iloc[:, 0].to_numpy(float)
    y = df.iloc[:, 1].to_numpy(float)
    order = np.argsort(x)
    x, y = x[order], normalize(y[order])
    label = TRACE_LABELS[i] if i < len(TRACE_LABELS) else Path(path).stem
    ax.plot(x, y, color=COLORS[i % len(COLORS)], linewidth=LINE_WIDTH,
            solid_capstyle="round", label=label)

ax.set_title(TITLE, fontsize=TITLE_SIZE, pad=10)
ax.set_xlabel(X_LABEL, fontsize=LABEL_SIZE, labelpad=6)
ax.set_ylabel(Y_LABEL, fontsize=LABEL_SIZE, labelpad=6)
ax.tick_params(labelsize=TICK_SIZE)
if len(DATA_PATHS) > 1 or TRACE_LABELS:
    ax.legend(fontsize=LEGEND_SIZE, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.margins(x=0.02)
fig.tight_layout()
if SAVE_AS:
    fig.savefig(SAVE_AS, dpi=300, bbox_inches="tight")
plt.show()
'''


# ----------------------------------------------------------------------
# Lifetime (decay + bi/tri-exponential fit with R^2)
# ----------------------------------------------------------------------
_LIFETIME_TEMPLATE = '''#!/usr/bin/env python3
"""
__TITLE__ — reproducible lifetime-decay plot for Google Colab.

HOW TO USE
----------
1. Upload your decay file to Colab and set DATA_PATH below.
2. Run the cell. The tail after the peak is fit with a bi- and a
   tri-exponential model; whichever fits better (by BIC) is kept, and the
   R^2, amplitudes (a_i), lifetimes (tau_i) and average tau are printed on
   the plot. Set FORCE_COMPONENTS = 3 to force a triexponential fit.
3. Edit the CONFIG block to restyle the figure.
"""

# ============================ CONFIG =================================
DATA_PATH = "INSERT DATA PATH HERE"

TITLE   = "__TITLE__"
X_LABEL = "__XLABEL__"
Y_LABEL = "__YLABEL__"

FORCE_COMPONENTS = None    # None = auto-pick 2 vs 3; or set 2 or 3

TITLE_SIZE  = 16
LABEL_SIZE  = 14
TICK_SIZE   = 11
LEGEND_SIZE = 9
ANNOT_SIZE  = 8

LINE_WIDTH  = 1.4
FIG_W, FIG_H = 6.0, 4.3
SAVE_AS = "lifetime.png"   # set to None to skip saving
# =====================================================================

__LOADER__

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

UNIT = "__UNIT__"


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

# Trim the flat pre-pulse baseline, then normalize.
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
ax.semilogy(x, np.clip(y, floor, None), color="#2E86C1",
            linewidth=LINE_WIDTH, label="Data")

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
    ax.semilogy(x[peak_idx:], np.clip(fit["yf"], floor, None), color="#333333",
                linewidth=1.1, linestyle="--", dashes=(4, 2), label=kind + " fit")
    lines = ["%s fit   R² = %.4f" % (kind, fit["r2"])]
    for i in range(n):
        lines.append("a%d = %.3f,  τ%d = %.2f %s  (%.0f%%)"
                     % (i+1, amps[i], i+1, taus[i], UNIT, w[i]*100))
    lines.append("c = %.3f" % c)
    lines.append("avg τ = %.2f %s" % (tau_avg, UNIT))
    ax.text(0.04, 0.04, "\\n".join(lines), transform=ax.transAxes,
            fontsize=ANNOT_SIZE, ha="left", va="bottom", linespacing=1.7)
    print("\\n".join(lines))

ax.set_ylim(bottom=floor*0.8, top=1.3)
ax.set_title(TITLE, fontsize=TITLE_SIZE, pad=10)
ax.set_xlabel(X_LABEL, fontsize=LABEL_SIZE, labelpad=6)
ax.set_ylabel(Y_LABEL, fontsize=LABEL_SIZE, labelpad=6)
ax.tick_params(labelsize=TICK_SIZE)
ax.legend(fontsize=LEGEND_SIZE, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
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


def make_colab_script(data_type, labels, unit=None):
    """Return a Colab-ready script string for a spectra data type."""
    if data_type == "lifetime":
        return _subst(_LIFETIME_TEMPLATE,
                      LOADER=_LOADER,
                      TITLE=labels["title"],
                      XLABEL=labels["xlabel"] if not unit else "Time (%s)" % unit,
                      YLABEL=labels["ylabel"],
                      UNIT=unit or "ns")
    # uvvis / emission
    return _subst(_XY_TEMPLATE,
                  LOADER=_LOADER,
                  TITLE=labels["title"],
                  XLABEL=labels["xlabel"],
                  YLABEL=labels["ylabel"])


def make_tem_colab_script():
    return _TEM_TEMPLATE
