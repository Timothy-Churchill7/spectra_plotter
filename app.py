#!/usr/bin/env python3
"""
Small Flask front-end for spectra_plotter.py.

Run:
    python app.py
Then open http://127.0.0.1:5000 and upload a .csv, .xlsx, or .txt file.
"""

import uuid
from pathlib import Path

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

from spectra_plotter import process_file

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".txt"}

TYPE_LABELS = {
    "uvvis": "UV-Vis Absorption",
    "emission": "Emission Spectrum",
    "lifetime": "Lifetime Decay",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return render_template("index.html", error="Please choose a file to upload."), 400

    filename = secure_filename(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return render_template(
            "index.html",
            error=f"Unsupported file type '{ext}'. Please upload a .csv, .xlsx, or .txt file.",
        ), 400

    job_id = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{job_id}{ext}"
    file.save(input_path)

    output_path = OUTPUT_DIR / f"{job_id}.png"
    try:
        _, data_type = process_file(str(input_path), str(output_path))
    except Exception as exc:
        return render_template(
            "index.html",
            error=f"Could not process '{filename}': {exc}",
        ), 400
    finally:
        input_path.unlink(missing_ok=True)

    image_url = url_for("static", filename=f"outputs/{job_id}.png")
    return render_template(
        "result.html",
        image_url=image_url,
        original_name=filename,
        type_label=TYPE_LABELS.get(data_type, data_type),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5050)
