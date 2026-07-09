# Plotter

A unified data-plotting web app that combines the **TEM Quantum Dot Analyzer**
and the **Spectra Plotter** into one site.

You start by choosing what kind of data you have:

| Type | What happens |
|------|--------------|
| **TEM** | Opens the TEM dot-analyzer workflow: upload a `.tif`, calibrate the scale bar, review/confirm detected dots, then download a **CSV** and a **size histogram**. |
| **Lifetime** | Upload a decay file → normalized semilog plot with an automatic **bi/tri-exponential fit** (R², amplitudes aᵢ, lifetimes τᵢ, avg τ). |
| **UV-Vis** | Upload one or more files → normalized xy absorption plot. Overlay multiple traces on one plot. |
| **Emission** | Upload one or more files → normalized xy emission plot. Overlay multiple traces on one plot. |

Every plot also comes with a **Google Colab-ready Python script** — it has an
`INSERT DATA PATH HERE` blank and a `CONFIG` block at the top with editable
titles, axis labels and font sizes, so you can restyle the figure yourself.

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
- `colab.py` — generates the downloadable Colab scripts.
- `templates/`, `static/` — landing, upload and result pages + styling.
- `samples/` — example data files (UV-Vis, emission, lifetime) for testing.

## Notes

- Accepted spectra formats: `.csv`, `.txt`, `.xlsx` (instrument metadata headers are skipped automatically).
- All spectra traces are peak-normalized (peak = 1.0).
- The lifetime fit keeps whichever of a bi- or tri-exponential model fits better (by BIC); set `FORCE_COMPONENTS = 3` in the Colab script to force a triexponential.
- Plotting uses matplotlib's object-oriented `Figure` API (no `pyplot` global state) so it's safe under the threaded dev server.
