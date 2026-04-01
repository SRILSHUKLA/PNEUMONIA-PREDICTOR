from pathlib import Path

from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

try:
    from .inference import FederatedMajorityEnsemble
except ImportError:
    from inference import FederatedMajorityEnsemble

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ensemble = FederatedMajorityEnsemble(repo_root=REPO_ROOT)


def _is_allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", result=None, error=None)

    uploaded = request.files.get("xray")
    if not uploaded or uploaded.filename == "":
        return render_template("index.html", result=None, error="Please upload an image file.")

    if not _is_allowed_file(uploaded.filename):
        return render_template(
            "index.html",
            result=None,
            error="Unsupported file format. Use PNG, JPG, JPEG, or WEBP.",
        )

    try:
        image = Image.open(uploaded.stream)
        result = ensemble.predict(image)
        return render_template("index.html", result=result, error=None)
    except UnidentifiedImageError:
        return render_template("index.html", result=None, error="Uploaded file is not a valid image.")
    except Exception as exc:
        return render_template("index.html", result=None, error=f"Prediction failed: {exc}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
