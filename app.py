"""
Airbag Kinematics Web Interface — Flask backend
================================================
Wraps the existing DeepLabV3+ / OCR / kinematics pipeline behind a web UI.

Endpoints
---------
GET  /                      -> main page (upload + live view + graphs)
POST /upload                -> accept a video, returns a job_id
GET  /stream/<job_id>       -> Server-Sent Events: progress + live frame previews
GET  /results/<job_id>      -> final kinematics JSON (arrays for graphs + summary)
GET  /download/<job_id>/csv -> the kinematics CSV
GET  /download/<job_id>/video -> annotated output mp4
GET  /download/<job_id>/plots -> matplotlib PNG
GET  /frame/<job_id>        -> latest annotated JPEG preview (polled fallback)

Run:  python app.py    (http://localhost:5000)

Notes
-----
* The heavy lifting (segmentation, OCR, kinematics) lives in analyzer.py, which
  is a refactor of your original script into a generator that yields progress.
* If the trained model / Tesseract / torch are missing, the app still runs and
  falls back to a DEMO analyzer so you can exercise the whole UI end-to-end.
"""

import os
import io
import json
import time
import uuid
import queue
import base64
import threading
import traceback

from flask import (
    Flask, request, jsonify, Response, send_file,
    render_template, abort, stream_with_context,
)

import analyzer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2 GB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# In-memory job registry. For a single-user local tool this is fine.
JOBS = {}  # job_id -> dict(state, queue, params, result, paths, latest_frame)


# --------------------------------------------------------------------------
# Job plumbing
# --------------------------------------------------------------------------
class Job:
    def __init__(self, job_id, video_path, params):
        self.id = job_id
        self.video_path = video_path
        self.params = params
        self.q = queue.Queue()          # progress events for SSE
        self.state = "queued"           # queued|running|done|error
        self.result = None              # final kinematics dict
        self.error = None
        self.latest_frame_jpeg = None   # bytes of most recent annotated preview
        self.out_csv = None
        self.out_video = None
        self.out_png = None
        self.thread = None

    def emit(self, event):
        """Push an event dict onto the SSE queue."""
        self.q.put(event)


def run_job(job: Job):
    job.state = "running"
    try:
        gen = analyzer.analyze(job.video_path, job.params, OUTPUT_DIR, job.id)
        for ev in gen:
            etype = ev.get("type")
            if etype == "frame":
                # Keep the latest preview around; also forward a lightweight
                # progress event (without the big base64 blob) every time.
                job.latest_frame_jpeg = ev.pop("_jpeg_bytes", None)
                job.emit(ev)
            elif etype == "done":
                job.result = ev["result"]
                job.out_csv = ev["result"]["paths"].get("csv")
                job.out_video = ev["result"]["paths"].get("video")
                job.out_png = ev["result"]["paths"].get("png")
                job.emit(ev)
            else:
                job.emit(ev)
        job.state = "done"
    except Exception as e:  # noqa
        job.state = "error"
        job.error = str(e)
        job.emit({"type": "error", "message": str(e),
                  "trace": traceback.format_exc()})
    finally:
        job.emit({"type": "_eos"})  # sentinel so the SSE loop can close


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html",
                           backend=analyzer.BACKEND_NAME,
                           demo=analyzer.IS_DEMO)


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify(error="No file part 'video'."), 400
    f = request.files["video"]
    if not f or f.filename == "":
        return jsonify(error="No file selected."), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify(error=f"Unsupported extension '{ext}'."), 400

    job_id = uuid.uuid4().hex[:12]
    save_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    f.save(save_path)

    # Create the job now but DON'T start analysis — wait for calibration.
    job = Job(job_id, save_path, params={})
    JOBS[job_id] = job

    # Extract first frame so the user can calibrate on it.
    frame_b64, w, h = analyzer.extract_first_frame(save_path)
    if frame_b64 is None:
        return jsonify(error="Could not read the first frame of the video."), 400

    return jsonify(job_id=job_id, width=w, height=h,
                   first_frame="data:image/jpeg;base64," + frame_b64)


@app.route("/analyze/<job_id>", methods=["POST"])
def analyze_start(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify(error="Unknown job. Re-upload the video."), 404
    if job.state != "queued":
        return jsonify(error=f"Job already {job.state}."), 409

    data = request.get_json(silent=True) or {}

    def _f(name, default):
        v = data.get(name)
        try:
            return float(v) if v not in (None, "") else default
        except (ValueError, TypeError):
            return default

    job.params = {
        "meters_per_pixel": _f("meters_per_pixel", 0.002459),
        "deployment_dir": data.get("deployment_dir") or "right-to-left",
        "steering_rim_y_frac": _f("steering_rim_y_frac", 0.844),
        "ocr_stride": int(_f("ocr_stride", 10)),
        "model_path": data.get("model_path") or "deeplabv3plus_airbag_best.pt",
        "preview_stride": int(_f("preview_stride", 3)),
    }

    job.thread = threading.Thread(target=run_job, args=(job,), daemon=True)
    job.thread.start()
    return jsonify(ok=True, job_id=job_id, params=job.params)


@app.route("/stream/<job_id>")
def stream(job_id):
    job = JOBS.get(job_id)
    if job is None:
        abort(404)

    @stream_with_context
    def event_source():
        # Replay nothing historical; just follow the queue live.
        while True:
            try:
                ev = job.q.get(timeout=30)
            except queue.Empty:
                # heartbeat to keep the connection alive through proxies
                yield ": keep-alive\n\n"
                continue
            if ev.get("type") == "_eos":
                break
            yield f"data: {json.dumps(ev)}\n\n"

    return Response(event_source(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/results/<job_id>")
def results(job_id):
    job = JOBS.get(job_id)
    if job is None:
        abort(404)
    if job.state == "error":
        return jsonify(error=job.error), 500
    if job.result is None:
        return jsonify(state=job.state), 202
    return jsonify(job.result)


@app.route("/frame/<job_id>")
def latest_frame(job_id):
    job = JOBS.get(job_id)
    if job is None or job.latest_frame_jpeg is None:
        abort(404)
    return Response(job.latest_frame_jpeg, mimetype="image/jpeg")


def _send(job_id, attr, mime, name):
    job = JOBS.get(job_id)
    if job is None:
        abort(404)
    path = getattr(job, attr)
    if not path or not os.path.exists(path):
        abort(404)
    return send_file(path, mimetype=mime, as_attachment=True,
                     download_name=name)


@app.route("/download/<job_id>/csv")
def dl_csv(job_id):
    return _send(job_id, "out_csv", "text/csv",
                 f"airbag_kinematics_{job_id}.csv")


@app.route("/download/<job_id>/video")
def dl_video(job_id):
    return _send(job_id, "out_video", "video/mp4",
                 f"airbag_analyzed_{job_id}.mp4")


@app.route("/download/<job_id>/plots")
def dl_png(job_id):
    return _send(job_id, "out_png", "image/png",
                 f"airbag_plots_{job_id}.png")


if __name__ == "__main__":
    print(f"Analyzer backend: {analyzer.BACKEND_NAME} "
          f"(demo={analyzer.IS_DEMO})")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
