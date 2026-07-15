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
import re
import time
import uuid
from pathlib import Path

from flask import (Flask, render_template, request, url_for,
                   send_from_directory, abort, redirect, session)
from werkzeug.utils import secure_filename

from spectra_core import (load_trace, render_session, DEFAULT_STYLE, LABELS,
                          FONT_FAMILIES, KIND_LABEL, TRACE_COLORS)
from colab import make_session_notebook
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
    for pattern in ("*.png", "*.py", "*.ipynb"):
        for f in OUTPUT_DIR.glob(pattern):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink(missing_ok=True)
            except OSError:
                pass


# ── Spectra session store ────────────────────────────────────────────────────
# {sid -> {mode, traces, style, png, script, result}}. In-memory (single worker).
SPECTRA_STORE = {}
LEGEND_LOCS = ["best", "upper right", "upper left", "lower right",
               "lower left", "center right", "center left"]


def _get_session():
    sid = session.get("spec_sid")
    return SPECTRA_STORE.get(sid) if sid else None


def _new_session(mode):
    sid = uuid.uuid4().hex
    session["spec_sid"] = sid
    SPECTRA_STORE[sid] = {"mode": mode, "traces": [], "style": dict(DEFAULT_STYLE),
                          "png": None, "script": None, "result": None}
    return SPECTRA_STORE[sid]


def _add_uploads(files, data_type, sess):
    """Save each upload to a temp file, load it as a trace, append to session."""
    for f in files:
        fname = secure_filename(f.filename) or "data"
        ext = Path(fname).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext}'. Upload .csv, .xlsx, or .txt.")
        tmp = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
        f.save(tmp)
        try:
            tr = load_trace(str(tmp), data_type, label=Path(fname).stem)
        finally:
            tmp.unlink(missing_ok=True)
        sess["traces"].append(tr)


def _render(sess):
    """Render the session's plot + Colab script into static/outputs, replacing
    the previous artifacts."""
    for key in ("png", "script"):
        if sess.get(key):
            (OUTPUT_DIR / sess[key]).unlink(missing_ok=True)
    job = uuid.uuid4().hex
    sess["result"] = render_session(sess, str(OUTPUT_DIR / f"{job}.png"))
    sess["png"] = f"{job}.png"
    (OUTPUT_DIR / f"{job}.ipynb").write_text(make_session_notebook(sess))
    sess["script"] = f"{job}.ipynb"


def _clamp_num(raw, lo, hi, fallback, cast=float):
    try:
        return max(lo, min(hi, cast(raw)))
    except (TypeError, ValueError):
        return fallback


def _apply_style_form(sess, form):
    s = sess["style"]
    for key in ("title", "xlabel", "ylabel"):
        s[key] = (form.get(key, "").strip() or None)

    fam = form.get("font_family", "DejaVu Sans")
    s["font_family"] = fam if fam in FONT_FAMILIES else "DejaVu Sans"

    for key, lo, hi in [("title_size", 6, 48), ("label_size", 6, 40),
                        ("tick_size", 4, 32), ("legend_size", 4, 32),
                        ("annot_size", 4, 24)]:
        s[key] = _clamp_num(form.get(key), lo, hi, s[key], cast=int)

    tc = form.get("text_color", "").strip()
    if re.fullmatch(r"#[0-9A-Fa-f]{6}", tc):
        s["text_color"] = tc

    s["line_width"] = _clamp_num(form.get("line_width"), 0.3, 6.0, s["line_width"])
    s["fig_w"] = _clamp_num(form.get("fig_w"), 3.0, 14.0, s["fig_w"])
    s["fig_h"] = _clamp_num(form.get("fig_h"), 2.0, 12.0, s["fig_h"])
    s["uvvis_min"] = _clamp_num(form.get("uvvis_min"), 0, 2000, s["uvvis_min"])
    s["uvvis_max"] = _clamp_num(form.get("uvvis_max"), 0, 2000, s["uvvis_max"])
    s["show_grid"] = form.get("show_grid") == "on"
    s["show_legend"] = form.get("show_legend") == "on"
    loc = form.get("legend_loc", "best")
    s["legend_loc"] = loc if loc in LEGEND_LOCS else "best"

    # Per-trace label + color overrides.
    for i, tr in enumerate(sess["traces"]):
        lbl = form.get(f"label_{i}", "").strip()
        if lbl:
            tr["label"] = lbl
        col = form.get(f"color_{i}", "").strip()
        if re.fullmatch(r"#[0-9A-Fa-f]{6}", col):
            tr["color"] = col


def _view_context(sess):
    res = sess.get("result") or {}
    fit_info = res.get("fit_info")
    meta = res.get("meta")
    style = sess["style"]

    # Effective per-trace color (falls back to the cycle) for swatches/inputs.
    traces_view = []
    for i, t in enumerate(sess["traces"]):
        traces_view.append({
            "index": i,
            "label": t["label"],
            "data_type": t["data_type"],
            "kind": KIND_LABEL[t["data_type"]],
            "color": t.get("color") or TRACE_COLORS[i % len(TRACE_COLORS)],
        })

    # Resolved (auto or overridden) title/labels — used as input placeholders.
    if sess["mode"] == "lifetime":
        unit = sess["traces"][0].get("unit") or "ns"
        resolved = {
            "title": style.get("title") or LABELS["lifetime"]["title"],
            "xlabel": style.get("xlabel") or f"Time ({unit})",
            "ylabel": style.get("ylabel") or LABELS["lifetime"]["ylabel"],
        }
    elif meta:
        resolved = {"title": meta["title"], "xlabel": meta["xlabel"], "ylabel": meta["ylabel"]}
    else:
        resolved = {"title": "", "xlabel": "", "ylabel": ""}

    return dict(
        sess=sess,
        style=style,
        traces_view=traces_view,
        mode=sess["mode"],
        image_url=url_for("static", filename=f"outputs/{sess['png']}") + f"?v={sess['png']}",
        script_url=url_for("download_script", job_id=Path(sess["script"]).stem),
        fit_info=fit_info,
        meta=meta,
        resolved=resolved,
        font_families=FONT_FAMILIES,
        legend_locs=LEGEND_LOCS,
        kind_label=KIND_LABEL,
    )


# ── Landing ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("landing.html")


# ── Spectra: initial upload from the landing page ────────────────────────────
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

    sess = _new_session("lifetime" if data_type == "lifetime" else "xy")
    try:
        _add_uploads(files, data_type, sess)
        _render(sess)
    except Exception as exc:  # noqa: BLE001 — surface parse/plot errors to the user
        session.pop("spec_sid", None)
        return render_template("upload.html", data_type=data_type, info=info,
                               error=f"Could not process your data: {exc}"), 400
    return redirect(url_for("spectra_view"))


# ── Spectra: the interactive result page ─────────────────────────────────────
@app.route("/spectra")
def spectra_view():
    sess = _get_session()
    if not sess or not sess["traces"]:
        return redirect(url_for("index"))
    return render_template("result.html", **_view_context(sess))


@app.route("/spectra/add", methods=["POST"])
def spectra_add():
    sess = _get_session()
    if not sess or sess["mode"] != "xy":
        return redirect(url_for("index"))
    data_type = request.form.get("data_type")
    if data_type not in ("uvvis", "emission"):
        return redirect(url_for("spectra_view"))
    files = [f for f in request.files.getlist("files") if f and f.filename]
    if not files:
        return redirect(url_for("spectra_view"))
    try:
        _add_uploads(files, data_type, sess)
        _render(sess)
    except Exception as exc:  # noqa: BLE001
        ctx = _view_context(sess)
        ctx["error"] = f"Could not add that data: {exc}"
        return render_template("result.html", **ctx), 400
    return redirect(url_for("spectra_view"))


@app.route("/spectra/style", methods=["POST"])
def spectra_style():
    sess = _get_session()
    if not sess or not sess["traces"]:
        return redirect(url_for("index"))
    _apply_style_form(sess, request.form)
    _render(sess)
    return redirect(url_for("spectra_view"))


@app.route("/spectra/remove", methods=["POST"])
def spectra_remove():
    sess = _get_session()
    if not sess:
        return redirect(url_for("index"))
    try:
        idx = int(request.form.get("index", -1))
    except (TypeError, ValueError):
        idx = -1
    if 0 <= idx < len(sess["traces"]):
        sess["traces"].pop(idx)
    if not sess["traces"]:
        return redirect(url_for("index"))
    _render(sess)
    return redirect(url_for("spectra_view"))


@app.route("/notebook/<job_id>.ipynb")
def download_script(job_id):
    # job_id is hex from uuid4; guard against path traversal.
    if not job_id.isalnum():
        abort(404)
    path = OUTPUT_DIR / f"{job_id}.ipynb"
    if not path.exists():
        abort(404)
    return send_from_directory(OUTPUT_DIR, f"{job_id}.ipynb", as_attachment=True,
                               download_name="plot_colab.ipynb",
                               mimetype="application/x-ipynb+json")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5100))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"\n  Plotter — open http://localhost:{port}\n")
    # use_reloader=False: the reloader forks a child process, which confuses
    # process-tracking launchers (and isn't needed for a normal run).
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
