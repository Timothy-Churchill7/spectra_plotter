#!/usr/bin/env python3
"""
Plotter — a unified data-plotting web app.

Landing page lets the user pick their data type:

    * TEM       → the TEM Quantum Dot Analyzer workflow (blueprint at /tem)
    * UV-Vis    → xy spectrum, overlay multiple traces
    * Emission  → xy spectrum, overlay multiple traces
    * Lifetime  → decay plot + bi/tri-exponential fit (R², coefficients)

For every generated plot the app also hands back a Colab-ready Python script
with an `INSERT DATA PATH HERE` blank and editable label/font-size settings.

Run:
    python app.py            (dev)
    gunicorn app:app         (prod)
"""

import os
import time
import uuid
from pathlib import Path

from flask import (Flask, render_template, request, url_for,
                   send_from_directory, abort, redirect)
from werkzeug.utils import secure_filename

from spectra_core import process_spectra, LABELS
from colab import make_colab_script
from tem_bp import tem_bp

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".txt"}
OUTPUT_MAX_AGE_SECONDS = 3600

# Data types handled by the upload-and-plot flow (TEM has its own blueprint).
SPECTRA_TYPES = {
    "uvvis":    {"label": "UV-Vis",   "multi": True,
                 "blurb": "Absorption spectra. Upload one file, or several to overlay."},
    "emission": {"label": "Emission", "multi": True,
                 "blurb": "Emission / PL spectra. Upload one file, or several to overlay."},
    "lifetime": {"label": "Lifetime", "multi": False,
                 "blurb": "TCSPC decay. Fit with a bi/tri-exponential model (R², τ values)."},
}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "plotter-dev-only")
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30 MB (TEM TIFs are large)
app.register_blueprint(tem_bp)


def _clean_old_outputs():
    cutoff = time.time() - OUTPUT_MAX_AGE_SECONDS
    for pattern in ("*.png", "*.py"):
        for f in OUTPUT_DIR.glob(pattern):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink(missing_ok=True)
            except OSError:
                pass


# ── Landing ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("landing.html")


# ── Spectra upload + plot ────────────────────────────────────────────────────
@app.route("/plot/<data_type>", methods=["GET"])
def plot_form(data_type):
    if data_type not in SPECTRA_TYPES:
        abort(404)
    return render_template("upload.html", data_type=data_type,
                           info=SPECTRA_TYPES[data_type])


@app.route("/plot/<data_type>", methods=["POST"])
def plot_run(data_type):
    if data_type not in SPECTRA_TYPES:
        abort(404)
    _clean_old_outputs()
    info = SPECTRA_TYPES[data_type]

    files = [f for f in request.files.getlist("files") if f and f.filename]
    if not files:
        return render_template("upload.html", data_type=data_type, info=info,
                               error="Please choose at least one data file."), 400
    if not info["multi"]:
        files = files[:1]

    job_id = uuid.uuid4().hex
    saved = []
    labels = []
    try:
        for f in files:
            fname = secure_filename(f.filename) or "data"
            ext = Path(fname).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file type '{ext}'. Upload .csv, .xlsx, or .txt.")
            p = UPLOAD_DIR / f"{job_id}_{len(saved)}{ext}"
            f.save(p)
            saved.append(p)
            labels.append(Path(fname).stem)

        out_png = OUTPUT_DIR / f"{job_id}.png"
        result = process_spectra([str(p) for p in saved], data_type, str(out_png),
                                 trace_labels=labels)

        # Write the matching Colab script next to the PNG.
        script = make_colab_script(data_type, result["labels"], unit=result["unit"])
        (OUTPUT_DIR / f"{job_id}.py").write_text(script)

        if result["fit_info"]:
            result["fit_info"]["unit"] = result["unit"]
    except Exception as exc:  # noqa: BLE001 — surface parse/plot errors to the user
        return render_template("upload.html", data_type=data_type, info=info,
                               error=f"Could not process your data: {exc}"), 400
    finally:
        for p in saved:
            p.unlink(missing_ok=True)

    return render_template(
        "result.html",
        data_type=data_type,
        info=info,
        image_url=url_for("static", filename=f"outputs/{job_id}.png"),
        script_url=url_for("download_script", job_id=job_id),
        source_names=[secure_filename(f.filename) for f in files],
        fit_info=result["fit_info"],
    )


@app.route("/script/<job_id>.py")
def download_script(job_id):
    # job_id is hex from uuid4; guard against path traversal.
    if not job_id.isalnum():
        abort(404)
    path = OUTPUT_DIR / f"{job_id}.py"
    if not path.exists():
        abort(404)
    return send_from_directory(OUTPUT_DIR, f"{job_id}.py", as_attachment=True,
                               download_name="plot_colab.py",
                               mimetype="text/x-python")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5100))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"\n  Plotter — open http://localhost:{port}\n")
    # use_reloader=False: the reloader forks a child process, which confuses
    # process-tracking launchers (and isn't needed for a normal run).
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
