

import os
import io
import re
import base64
import numpy as np

# ---- Optional heavy deps. We degrade gracefully if any are missing. ----
_HAVE_CV2 = _HAVE_TORCH = _HAVE_SMP = _HAVE_TESS = _HAVE_SCIPY = False
try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    pass
try:
    import torch
    _HAVE_TORCH = True
except Exception:
    pass
try:
    import segmentation_models_pytorch as smp
    _HAVE_SMP = True
except Exception:
    pass
try:
    import pytesseract
    _HAVE_TESS = True
except Exception:
    pass
try:
    from scipy.signal import savgol_filter
    _HAVE_SCIPY = True
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _real_available(model_path):
    return (_HAVE_CV2 and _HAVE_TORCH and _HAVE_SMP
            and os.path.exists(model_path))


def extract_first_frame(video_path, max_w=900):
    """Return (base64_jpeg, width, height) of the video's first frame.

    Used by the UI so the user can click calibration points on it. Falls back
    to a synthetic placeholder frame when OpenCV is unavailable (demo mode).
    """
    if _HAVE_CV2:
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None, 0, 0
        h, w = frame.shape[:2]
        # Encode at native resolution; the browser scales for display but we
        # send true pixel dims so calibration maps back to full-res pixels.
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            return None, 0, 0
        return base64.b64encode(buf.tobytes()).decode(), int(w), int(h)

    # Demo placeholder: a synthetic 512x384 dashboard-ish frame.
    w, h = 512, 384
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("#202830"); ax.axis("off")
    ax.add_patch(plt.Rectangle((0.30, 0.78), 0.40, 0.16, color="#0b0e12"))
    ax.text(0.5, 0.86, "+000.0 msec", color="#cfe", ha="center", va="center",
            fontsize=12, family="monospace", transform=ax.transAxes)
    ax.add_patch(plt.Circle((0.5, 0.40), 0.18, fill=False, ec="#888", lw=2))
    ax.text(0.5, 0.06, "DEMO FRAME — calibrate here", color="#9fb0c0",
            ha="center", fontsize=9, transform=ax.transAxes)
    buf = io.BytesIO(); fig.savefig(buf, format="jpg"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode(), w, h


# Decide backend lazily at import for the banner; per-job re-checked too.
_DEFAULT_MODEL = "deeplabv3plus_airbag_best.pt"
IS_DEMO = not _real_available(_DEFAULT_MODEL)
BACKEND_NAME = "DeepLabV3+ (real)" if not IS_DEMO else "DEMO (synthetic)"


# ==========================================================================
# Shared kinematics math (same logic as the original script)
# ==========================================================================
def _safe_savgol(series, window_len, poly_order=3):
    series = np.asarray(series, dtype=float)
    if not _HAVE_SCIPY or window_len < poly_order + 2 or len(series) < window_len:
        return series
    return savgol_filter(series, window_len, poly_order)


def _compute_kinematics(frames, times, lead_px, lowest_px, area_px,
                        cx, cy, params, fallback_fps):
    """Return the results dict shared by real + demo paths."""
    frames = np.asarray(frames, dtype=float)
    lead_px = np.asarray(lead_px, dtype=float)
    lowest_px = np.asarray(lowest_px, dtype=float)
    area_px = np.asarray(area_px, dtype=float)
    raw_times = np.asarray(times, dtype=float)

    mpp = params["meters_per_pixel"]
    direction = params["deployment_dir"]
    rim_y = params["_steering_rim_y_px"]

    # --- Build a trustworthy time axis from sampled OCR reads ---
    valid_mask = ~np.isnan(raw_times)
    n_valid = int(valid_mask.sum())
    MIN_OCR_READS = 10
    MAX_FIT_RMS_SEC = 0.050
    OUTLIER_SIGMA = 3.0
    fit_slope = None
    use_fallback = False
    notes = []

    if n_valid >= MIN_OCR_READS:
        fv, tv = frames[valid_mask], raw_times[valid_mask]
        m, b = np.polyfit(fv, tv, 1)
        resid = tv - (m * fv + b)
        rms = float(np.sqrt(np.mean(resid ** 2)))
        if rms > 0:
            keep = np.abs(resid) < OUTLIER_SIGMA * rms
            if keep.sum() >= MIN_OCR_READS and keep.sum() < len(fv):
                m, b = np.polyfit(fv[keep], tv[keep], 1)
                r2 = tv[keep] - (m * fv[keep] + b)
                rms = float(np.sqrt(np.mean(r2 ** 2)))
        if m <= 0 or rms > MAX_FIT_RMS_SEC:
            use_fallback = True
            notes.append(f"OCR fit rejected (slope={m:.5f}, rms={rms*1000:.1f}ms); using video FPS.")
        else:
            fit_slope = m
            timestamps = m * frames + b
            notes.append(f"OCR-derived effective FPS = {1.0/m:.2f}")
    else:
        use_fallback = True
        notes.append(f"Only {n_valid} OCR reads (<{MIN_OCR_READS}); using video FPS.")

    if use_fallback:
        timestamps = frames / fallback_fps

    timestamps = timestamps - timestamps[0]

    # --- Horizontal displacement / velocity ---
    init = lead_px[0]
    if direction in ("right-to-left", "bottom-to-top"):
        disp_px = init - lead_px
    else:
        disp_px = lead_px - init
    disp_m = disp_px * mpp
    area_m2 = area_px * (mpp ** 2)

    # --- Vertical drop past steering rim ---
    drop_px = lowest_px - rim_y
    drop_m = drop_px * mpp

    n = len(disp_m)
    window = min(11, n if n % 2 == 1 else n - 1)
    sm_disp = _safe_savgol(disp_m, window)
    sm_area = _safe_savgol(area_m2, window)
    sm_drop = _safe_savgol(drop_m, window)
    vel = np.gradient(sm_disp, timestamps)
    sm_vel = _safe_savgol(vel, window)

    # --- Velocity vs. Displacement (phase) curve ---
    # Displacement rises through the deployment, then plateaus/repeats once
    # the airbag stops inflating (or oscillates). Past that point the curve
    # would double back on itself on a v-vs-x plot, so we cut at the first
    # point displacement reaches its running maximum.
    if len(sm_disp):
        phase_cutoff = int(np.argmax(sm_disp)) + 1  # inclusive of the peak
    else:
        phase_cutoff = 0
    phase_disp = sm_disp[:phase_cutoff]
    phase_vel = sm_vel[:phase_cutoff]

    df = pd.DataFrame({
        "Frame_ID": frames.astype(int),
        "True_Video_Time_ms": timestamps * 1000.0,
        "Displacement_mm": sm_disp * 1000.0,
        "Velocity_mm_per_s": sm_vel * 1000.0,
        # negative = below the steering rim (curve dips down)
        "Drop_Past_Rim_mm": -sm_drop * 1000.0,
        "Surface_Area_sq_mm": sm_area * 1e6,
        "Raw_Centroid_X_Px": np.asarray(cx, dtype=float),
        "Raw_Centroid_Y_Px": np.asarray(cy, dtype=float),
        "Raw_Lowest_Y_Px": lowest_px,
    })

    summary = {
        "peak_velocity_mps": float(np.nanmax(sm_vel)),
        "total_displacement_mm": float(sm_disp[-1] * 1000),
        "max_drop_below_rim_mm": float(np.nanmax(sm_drop) * 1000),
        "duration_ms": float(timestamps[-1] * 1000),
        "time_source": ("OCR" if fit_slope else "video FPS fallback"),
        "effective_fps": (float(1.0 / fit_slope) if fit_slope else float(fallback_fps)),
        "valid_frames": int(n),
    }

    series = {
        "time_sec": timestamps.tolist(),
        "displacement_m": sm_disp.tolist(),
        "displacement_raw_m": disp_m.tolist(),
        "velocity_mps": sm_vel.tolist(),
        "drop_below_rim_m": sm_drop.tolist(),
        "area_m2": sm_area.tolist(),
        "phase_displacement_m": phase_disp.tolist(),
        "phase_velocity_mps": phase_vel.tolist(),
    }
    return df, series, summary, notes


def _render_plots(series, png_path):
    t = np.asarray(series["time_sec"]) * 1000.0           # s -> ms
    disp_mm = np.asarray(series["displacement_m"]) * 1000.0
    disp_raw_mm = np.asarray(series["displacement_raw_m"]) * 1000.0
    vel_mmps = np.asarray(series["velocity_mps"]) * 1000.0
    # flip sign: drop BELOW the rim is negative (curve dips down)
    drop_mm = -np.asarray(series["drop_below_rim_m"]) * 1000.0
    area_mm2 = np.asarray(series["area_m2"]) * 1e6
    phase_disp_mm = np.asarray(series.get("phase_displacement_m", [])) * 1000.0
    phase_vel_mmps = np.asarray(series.get("phase_velocity_mps", [])) * 1000.0

    fig, axes = plt.subplots(5, 1, figsize=(12, 13.5))

    ax = axes[0]
    ax.plot(t, disp_mm, color="blue", lw=2, label="Filtered")
    ax.scatter(t, disp_raw_mm, color="lightskyblue", s=10, alpha=0.5, label="Raw")
    ax.set_ylabel("Displacement (mm)")
    ax.set_title("Airbag Deployment Kinematics Analysis Profiles")
    ax.grid(True, ls="--"); ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot(t, vel_mmps, color="red", lw=2)
    ax.set_ylabel("Velocity (mm/s)"); ax.grid(True, ls="--")

    ax = axes[2]
    ax.plot(t, drop_mm, color="orange", lw=2)
    # highlighted zero line = steering rim
    ax.axhline(0, color="#0A4DA8", ls="--", lw=2, label="Steering Rim (0)")
    ax.fill_between(t, drop_mm, 0, where=(drop_mm < 0), color="orange",
                    alpha=0.3, label="Airbag Below Rim")
    ax.set_ylabel("Vertical Drop Past Rim (mm)")
    ax.legend(loc="lower left"); ax.grid(True, ls="--")

    ax = axes[3]
    ax.plot(t, area_mm2, color="purple", lw=2)
    ax.set_xlabel("True Elapsed Time (ms)")
    ax.set_ylabel("Fabric Area (mm\u00b2)"); ax.grid(True, ls="--")

    ax = axes[4]
    ax.plot(phase_disp_mm, phase_vel_mmps, color="teal", lw=2)
    ax.set_xlabel("Displacement (mm)")
    ax.set_ylabel("Velocity (mm/s)")
    ax.set_title("Velocity vs. Displacement (up to peak displacement)", fontsize=10)
    ax.grid(True, ls="--")

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)


# ==========================================================================
# REAL pipeline
# ==========================================================================
def _analyze_real(video_path, params, output_dir, job_id):
    import albumentations as A

    device = "cuda" if torch.cuda.is_available() else "cpu"
    yield {"type": "log", "message": f"Using device: {device.upper()}"}

    model_path = params["model_path"]
    yield {"type": "log", "message": "Loading DeepLabV3+ model..."}
    model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None,
                              in_channels=3, classes=1, activation=None)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    model.to(device).eval()

    if _HAVE_TESS:
        # Allow an override via params; otherwise use the standard Windows path.
        pytesseract.pytesseract.tesseract_cmd = (
            params.get("tesseract_cmd")
            or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    yield {"type": "meta", "width": width, "height": height,
           "total_frames": total, "fps": float(fps)}

    mpp = params["meters_per_pixel"]
    direction = params["deployment_dir"]
    rim_y = int(params["steering_rim_y_frac"] * height)
    params["_steering_rim_y_px"] = rim_y
    min_area_px = 5e-5 * (width * height)
    LOWEST_Y_PCTL = 98
    OCR_STRIDE = params["ocr_stride"]
    preview_stride = max(1, params.get("preview_stride", 3))

    TIMER_Y1, TIMER_Y2 = int(0.010 * height), int(0.060 * height)
    TIMER_X1, TIMER_X2 = int(0.32 * width), int(0.56 * height and width * 0.56)
    TIMER_X2 = int(0.56 * width)

    out_video = os.path.join(output_dir, f"airbag_analyzed_{job_id}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))
    resize = A.Compose([A.Resize(512, 512)])

    lead, cxs, cys, lowys, areas, idxs, tstamps = [], [], [], [], [], [], []
    last_good = None
    fcount = 0

    def ocr_timer(crop, last):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        cfg = r"--psm 7 -c tessedit_char_whitelist=0123456789:.+-"
        try:
            txt = pytesseract.image_to_string(th, config=cfg).strip() if _HAVE_TESS else ""
        except Exception:
            txt = ""   # Tesseract binary missing/unreachable -> fall back to video FPS
        m = re.search(r"[-+]?\d+(?:\.\d+)?", txt)
        if not m:
            return np.nan
        try:
            t = float(m.group()) / 1000.0
        except ValueError:
            return np.nan
        if last is not None and t + 1e-6 < last:
            return np.nan
        return t

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            fcount += 1

            do_ocr = (fcount == 1) or (fcount % OCR_STRIDE == 0)
            if do_ocr:
                crop = frame[TIMER_Y1:TIMER_Y2, TIMER_X1:TIMER_X2]
                cur_t = ocr_timer(crop, last_good)
                if not np.isnan(cur_t):
                    last_good = cur_t
            else:
                cur_t = np.nan

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = resize(image=rgb)["image"]
            tens = torch.from_numpy(res.transpose(2, 0, 1)).float() / 255.0
            tens = tens.unsqueeze(0).to(device)
            logits = model(tens)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).cpu().numpy().squeeze().astype(np.uint8)
            mask = cv2.resize(pred, (width, height),
                              interpolation=cv2.INTER_NEAREST)

            proc = mask * 255
            k = np.ones((7, 7), np.uint8)
            clean = cv2.morphologyEx(proc, cv2.MORPH_OPEN, k)
            cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
            cx = cy = le = lowy = None
            area = 0
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                area = cv2.contourArea(c)
                if area > min_area_px:
                    M = cv2.moments(c)
                    if M["m00"]:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    pts = c.reshape(-1, 2)
                    xs, ys = pts[:, 0], pts[:, 1]
                    if direction == "left-to-right":
                        le = int(np.percentile(xs, 98))
                    elif direction == "right-to-left":
                        le = int(np.percentile(xs, 2))
                    elif direction == "bottom-to-top":
                        le = int(np.percentile(ys, 2))
                    elif direction == "top-to-bottom":
                        le = int(np.percentile(ys, 98))
                    else:
                        le = int(np.percentile(xs, 98))
                    lowy = int(np.percentile(ys, LOWEST_Y_PCTL))

            if le is not None and cx is not None and lowy is not None:
                lead.append(le); cxs.append(cx); cys.append(cy)
                lowys.append(lowy); areas.append(area)
                idxs.append(fcount); tstamps.append(cur_t)

            # overlay
            ov = np.zeros_like(frame); ov[:] = [0, 0, 255]
            hi = cv2.bitwise_and(ov, ov, mask=mask)
            blend = cv2.addWeighted(frame, 0.70, hi, 0.30, 0)
            cv2.line(blend, (0, rim_y), (width, rim_y), (255, 150, 0), 1)
            if cx is not None:
                cv2.circle(blend, (cx, cy), 6, (0, 255, 0), -1)
                if direction in ("left-to-right", "right-to-left"):
                    cv2.line(blend, (le, 0), (le, height), (0, 255, 255), 2)
                else:
                    cv2.line(blend, (0, le), (width, le), (0, 255, 255), 2)
            drop_mm = None
            if lowy is not None:
                cv2.line(blend, (0, lowy), (width, lowy), (0, 165, 255), 1)
                d = lowy - rim_y
                if d > 0:
                    drop_mm = d * mpp * 1000
                    cv2.putText(blend, f"Drop: {drop_mm:.1f} mm",
                                (10, min(lowy + 18, height - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            writer.write(blend)

            if fcount % preview_stride == 0 or fcount == total:
                small = cv2.resize(blend, (min(640, width),
                                           int(height * min(640, width) / width)))
                ok, buf = cv2.imencode(".jpg", small,
                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
                jpeg = buf.tobytes() if ok else None
                yield {
                    "type": "frame", "frame": fcount, "total": total,
                    "pct": round(100 * fcount / max(total, 1), 1),
                    "time_sec": (None if np.isnan(cur_t) else round(float(cur_t), 4)),
                    "drop_mm": (None if drop_mm is None else round(drop_mm, 1)),
                    "image": "data:image/jpeg;base64," +
                             (base64.b64encode(jpeg).decode() if jpeg else ""),
                    "_jpeg_bytes": jpeg,
                }

    cap.release(); writer.release()

    if len(idxs) < 15:
        raise RuntimeError("Too few frames contained an airbag silhouette "
                           "to compute kinematics safely.")

    df, series, summary, notes = _compute_kinematics(
        idxs, tstamps, lead, lowys, areas, cxs, cys, params, fps)
    for nm in notes:
        yield {"type": "log", "message": nm}

    csv_path = os.path.join(output_dir, f"airbag_kinematics_{job_id}.csv")
    df.to_csv(csv_path, index=False)
    png_path = os.path.join(output_dir, f"airbag_plots_{job_id}.png")
    _render_plots(series, png_path)

    yield {"type": "done", "result": {
        "summary": summary, "series": series,
        "paths": {"csv": csv_path, "video": out_video, "png": png_path},
        "backend": "DeepLabV3+ (real)"}}


# ==========================================================================
# DEMO pipeline — synthesizes a plausible deployment so the UI fully works
# ==========================================================================
def _analyze_demo(video_path, params, output_dir, job_id):
    yield {"type": "log",
           "message": "Real backend unavailable (torch / model / opencv "
                      "missing). Running DEMO analyzer on synthetic physics."}

    width, height, total, fps = 512, 384, 160, 1000.0  # high-speed cam
    params["_steering_rim_y_px"] = int(params["steering_rim_y_frac"] * height)
    rim_y = params["_steering_rim_y_px"]
    yield {"type": "meta", "width": width, "height": height,
           "total_frames": total, "fps": float(fps)}

    preview_stride = max(1, params.get("preview_stride", 3))
    mpp = params["meters_per_pixel"]

    # Synthetic ground-truth deployment kinematics
    idxs, tstamps, lead, lowys, areas, cxs, cys = [], [], [], [], [], [], []
    start = 20  # airbag becomes visible at frame 20
    init_le = 430.0
    for f in range(1, total + 1):
        if f < start:
            continue
        tt = (f - start) / fps  # seconds since first detection
        # logistic-ish inflation: leading edge sweeps right-to-left
        prog = 1.0 / (1.0 + np.exp(-12 * (tt - 0.045)))
        le = init_le - 360 * prog + np.random.normal(0, 1.5)
        low = rim_y - 30 + 120 * prog + np.random.normal(0, 1.2)
        area = 1500 + 52000 * prog + np.random.normal(0, 600)
        cx = int(init_le - 180 * prog)
        cy = int(rim_y - 40 + 60 * prog)
        idxs.append(f)
        # emulate OCR every Nth frame with small noise; NaN otherwise
        if (f == start) or (f % params["ocr_stride"] == 0):
            tstamps.append(tt + np.random.normal(0, 0.0008))
        else:
            tstamps.append(np.nan)
        lead.append(le); lowys.append(low); areas.append(max(area, 1))
        cxs.append(cx); cys.append(cy)

        if f % (preview_stride * 2) == 0 or f == total:
            # draw a simple synthetic preview frame
            img = np.full((height, width, 3), 25, np.uint8)
            cv = _np_cv()  # tiny shim so demo works without cv2 too
            if cv is not None:
                cv.line(img, (0, rim_y), (width, rim_y), (255, 150, 0), 1)
                bag_r = int(10 + 90 * prog)
                cv.circle(img, (cx, int(rim_y - 10 + 40 * prog)),
                          bag_r, (0, 0, 200), -1)
                cv.circle(img, (cx, int(rim_y - 10 + 40 * prog)),
                          bag_r, (0, 0, 255), 2)
                cv.line(img, (int(le), 0), (int(le), height), (0, 255, 255), 2)
                cv.putText(img, f"DEMO  t={tt*1000:5.1f} ms",
                           (8, 22), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                           (255, 255, 255), 1)
                ok, buf = cv.imencode(".jpg", img,
                                      [cv.IMWRITE_JPEG_QUALITY, 70])
                jpeg = buf.tobytes() if ok else None
            else:
                jpeg = _png_placeholder(tt)
            drop_mm = max(0.0, (low - rim_y) * mpp * 1000)
            yield {
                "type": "frame", "frame": f, "total": total,
                "pct": round(100 * f / total, 1),
                "time_sec": round(tt, 4),
                "drop_mm": round(drop_mm, 1),
                "image": "data:image/jpeg;base64," +
                         (base64.b64encode(jpeg).decode() if jpeg else ""),
                "_jpeg_bytes": jpeg,
            }

    df, series, summary, notes = _compute_kinematics(
        idxs, tstamps, lead, lowys, areas, cxs, cys, params, fps)
    summary["backend"] = "DEMO"
    for nm in notes:
        yield {"type": "log", "message": nm}

    csv_path = os.path.join(output_dir, f"airbag_kinematics_{job_id}.csv")
    df.to_csv(csv_path, index=False)
    png_path = os.path.join(output_dir, f"airbag_plots_{job_id}.png")
    _render_plots(series, png_path)

    yield {"type": "done", "result": {
        "summary": summary, "series": series,
        "paths": {"csv": csv_path, "video": None, "png": png_path},
        "backend": "DEMO (synthetic)"}}


def _np_cv():
    try:
        import cv2
        return cv2
    except Exception:
        return None


def _png_placeholder(tt):
    # Minimal grey JPEG via matplotlib if cv2 isn't present.
    fig = plt.figure(figsize=(4, 3), dpi=80)
    fig.text(0.5, 0.5, f"DEMO  t={tt*1000:.1f} ms", ha="center", va="center")
    buf = io.BytesIO()
    fig.savefig(buf, format="jpg")
    plt.close(fig)
    return buf.getvalue()


# ==========================================================================
# Public entry point
# ==========================================================================
def analyze(video_path, params, output_dir, job_id):
    if _real_available(params.get("model_path", _DEFAULT_MODEL)):
        yield from _analyze_real(video_path, params, output_dir, job_id)
    else:
        yield from _analyze_demo(video_path, params, output_dir, job_id)
