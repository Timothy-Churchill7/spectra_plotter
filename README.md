# Plotter

A unified data-plotting web app that combines the **TEM Quantum Dot Analyzer**
and the **Spectra Plotter** into one site.

You start by choosing what kind of data you have:

| Type | What happens |
|------|--------------|
| **TEM** | Opens the TEM dot-analyzer workflow: upload a `.tif`, calibrate the scale bar, review/confirm detected dots, then download a **CSV** and a **size histogram**. |
| **Lifetime** | Upload a decay file → normalized semilog plot with an automatic **bi/tri-exponential fit** (R², amplitudes aᵢ, lifetimes τᵢ, avg τ). |
| **UV-Vis** | Upload file(s) → normalized xy absorption plot, cropped to 300–700 nm. |
| **Emission** | Upload file(s) → normalized xy emission plot. |

### Building up a spectra plot

After the first UV-Vis / emission plot, the result page lets you **"Plot more
data on the same graph."** Each time you add data you pick its type, so you can:

- overlay several samples of the **same kind** (each labeled by file name), and
- mix **emission and absorbance on one plot** — the legend annotates each trace
  with its kind (e.g. `Sample 1 (Emission)`, `Sample 2 (Absorbance)`).

UV-Vis absorbance traces are cropped to, and normalized within, a **300–700 nm**
window (editable in the styling panel).

### Styling tool

Every spectra/lifetime result has a **Customize appearance** panel: edit the
title and axis labels, choose a font, set title/axis/tick/legend font sizes,
pick a text color, line width and figure size, toggle the legend/grid, and
rename or recolor each individual trace (or remove it). The plot re-renders
server-side on **Apply**.

Every plot also comes with a **Google Colab notebook (`.ipynb`)**. It opens with
a cell that mounts your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

followed by a **Config** cell (the `INSERT DATA PATH HERE` blanks plus constants
that mirror the styling tool — titles, labels, fonts, sizes, colors, the UV-Vis
window, and each trace's type/label/color) and the plotting cells. So you point
the paths at files in your Drive, run, and reproduce the exact styled figure.

## Run it

```bash
cd Plotter
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python app.py           # http://localhost:5000
```

For production: `gunicorn app:app`.

## Layout

- `app.py` — landing page (type picker) + UV-Vis / emission / lifetime upload & plot routes.
- `spectra_core.py` — file parsing + plotting for the three spectra types.
- `tem_bp.py` — the TEM analyzer, as a Flask blueprint mounted at `/tem`, plus the size histogram.
- `tem_dot_analyzer.py` — dot-detection / scale-bar logic (unchanged from the original tool).
- `colab.py` — generates the downloadable Colab notebooks (`.ipynb`).
- `templates/`, `static/` — landing, upload and result pages + styling.
- `samples/` — example data files (UV-Vis, emission, lifetime) for testing.

## Notes

- Accepted spectra formats: `.csv`, `.txt`, `.xlsx` (instrument metadata headers are skipped automatically).
- All spectra traces are peak-normalized (peak = 1.0).
- The lifetime fit keeps whichever of a bi- or tri-exponential model fits better (by BIC); set `FORCE_COMPONENTS = 3` in the Colab notebook's Config cell to force a triexponential.
- Plotting uses matplotlib's object-oriented `Figure` API (no `pyplot` global state) so it's safe under the threaded dev server.
