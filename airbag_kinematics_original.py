import os
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
import segmentation_models_pytorch as smp
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pytesseract
import re

# ==========================================
# 1. SETTINGS & CALIBRATION CONSTANTS
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")

# File Paths
INPUT_VIDEO_PATH = "airbag_deployment3.mp4"
OUTPUT_VIDEO_PATH = "airbag_analyzed_output.mp4"
OUTPUT_CSV_PATH = "airbag_kinematics_results.csv"
MODEL_PATH = "deeplabv3plus_airbag_best.pt"

# --- Tesseract Configuration ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\rishik.yadav\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# --- Timer ROI Coordinates ---
# Timer "+NNN.N msec" sits in the top black banner of the Autoliv overlay.
# Defined as FRACTIONS of the actual frame so it works at any resolution.
TIMER_Y_FRAC = (0.010, 0.060)
TIMER_X_FRAC = (0.32,  0.56)

# --- OCR controls ---
OCR_STRIDE = 10                # OCR one frame in N (camera clock is linear; sampling is enough)
OCR_DEBUG_EVERY = 100          # save a debug crop every N frames (set to 0 to disable)
MIN_OCR_READS = 10             # need at least this many valid reads to trust the fit
MAX_FIT_RMS_SEC = 0.050        # >50 ms RMS scatter from the line = OCR is garbage
OUTLIER_REJECT_SIGMA = 3.0     # drop reads more than this many residual-sigmas from the fit

# --- Physical Setup Calibration ---
METERS_PER_PIXEL = 0.002459      # measured: 0.100 m / pixel_distance between marks
DEPLOYMENT_DIR = "right-to-left"  # left-to-right, right-to-left, top-to-bottom, bottom-to-top

# --- Steering Rim Reference (for vertical-drop analysis) ---
# A fixed horizontal line representing the steering wheel rim. We measure how
# far the airbag's lowest point drops BELOW this line during deployment.
# Defined as a fraction of frame height so it scales with resolution. On a
# 512x384 frame the historical value of y=324 corresponds to fraction 0.844.
STEERING_RIM_Y_FRAC = 0.844

# --- Area filter (kept fractional so it scales with resolution) ---
MIN_AREA_FRAC = 5e-5

# --- Lowest-Y percentile (robust against single-pixel noise) ---
# Use a high percentile of contour y-values instead of the absolute max.
# 98 = "almost the lowest point" but ignores stray pixels.
LOWEST_Y_PCTL = 98

if not os.path.exists(INPUT_VIDEO_PATH):
    print(f"Error: Please place your video file at '{INPUT_VIDEO_PATH}' before running.")
    exit()

# ==========================================
# 2. LOAD DEEPLABV3+ TRAINED MODEL
# ==========================================
print("Loading DeepLabV3+ model framework...")
model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

# ==========================================
# 3. OPEN VIDEO STREAM & INITIALIZE WRITER
# ==========================================
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

# Visual playback FPS for the OUTPUT mp4. Kinematics math uses OCR timestamps.
visual_fps = cap.get(cv2.CAP_PROP_FPS)
if not visual_fps or visual_fps <= 0:
    visual_fps = 30.0  # playback fallback

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_area = float(width * height)
min_area_px = MIN_AREA_FRAC * frame_area

print(f"Video resolution: {width} x {height}")

# Resolve fractional ROI to actual pixel coordinates
TIMER_Y1 = int(TIMER_Y_FRAC[0] * height)
TIMER_Y2 = int(TIMER_Y_FRAC[1] * height)
TIMER_X1 = int(TIMER_X_FRAC[0] * width)
TIMER_X2 = int(TIMER_X_FRAC[1] * width)
print(f"Timer ROI: x=[{TIMER_X1},{TIMER_X2}]  y=[{TIMER_Y1},{TIMER_Y2}]")

# Resolve steering-rim fraction to a pixel row
STEERING_RIM_Y = int(STEERING_RIM_Y_FRAC * height)
print(f"Steering rim reference line: y={STEERING_RIM_Y}")

# Sanity-check ROI is inside the frame BEFORE we start a long run
assert 0 <= TIMER_Y1 < TIMER_Y2 <= height, "TIMER_Y ROI out of bounds for this video"
assert 0 <= TIMER_X1 < TIMER_X2 <= width,  "TIMER_X ROI out of bounds for this video"
assert 0 <= STEERING_RIM_Y < height,       "STEERING_RIM_Y out of bounds for this video"

# Optional: dump the first frame with the ROI drawn so you can visually verify
# the timer crop and rim line before the full run.
_ok, _first = cap.read()
if _ok:
    _vis = _first.copy()
    cv2.rectangle(_vis, (TIMER_X1, TIMER_Y1), (TIMER_X2, TIMER_Y2), (0, 0, 255), 2)
    cv2.line(_vis, (0, STEERING_RIM_Y), (width, STEERING_RIM_Y), (255, 150, 0), 1)
    cv2.imwrite("debug_setup.jpg", _vis)
    print("Wrote debug_setup.jpg — confirm the timer box and rim line are correctly placed.")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind to start

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, visual_fps, (width, height))
print(f"Note: output mp4 plays at {visual_fps:.1f} fps for viewing; "
      f"true event timing comes from OCR.")

resize_transform = A.Compose([A.Resize(512, 512)])

# In-memory arrays
raw_leading_edges_px = []
raw_centroids_x = []
raw_centroids_y = []
raw_lowest_y_px = []         # bottom edge of airbag (for vertical-drop analysis)
raw_areas_px = []
valid_frame_indices = []
raw_timestamps_sec = []

print(f"Processing sequence ({total_frames} frames total) for physics extraction...")


# ------------------------------------------
# Helper: OCR a timer crop and return seconds (or NaN on failure)
# ------------------------------------------
def ocr_timer(timer_crop, last_good_time):
    gray = cv2.cvtColor(timer_crop, cv2.COLOR_BGR2GRAY)
    # Otsu auto-adapts to the overlay's brightness.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Whitelist includes +- because the timer reads "+NNN.N msec".
    custom_config = r'--psm 7 -c tessedit_char_whitelist=0123456789:.+-'
    text = pytesseract.image_to_string(thresh, config=custom_config).strip()

    m = re.search(r'[-+]?\d+(?:\.\d+)?', text)
    if not m:
        return np.nan, thresh
    try:
        # Timer is in MILLISECONDS (e.g. "+088.0 msec") -> convert to seconds.
        t = float(m.group()) / 1000.0
    except ValueError:
        return np.nan, thresh

    if last_good_time is not None and t + 1e-6 < last_good_time:
        return np.nan, thresh
    return t, thresh


# ==========================================
# 4. INFERENCE, OCR & KINEMATICS LOOP
# ==========================================
frame_count = 0
last_good_ocr = None

with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # --- 1. OCR Timer Extraction (sampled, not every frame) ---
        do_ocr = (frame_count == 1) or (frame_count % OCR_STRIDE == 0)
        if do_ocr:
            timer_crop = frame[TIMER_Y1:TIMER_Y2, TIMER_X1:TIMER_X2]
            current_time_sec, thresh_crop = ocr_timer(timer_crop, last_good_ocr)
            if not np.isnan(current_time_sec):
                last_good_ocr = current_time_sec
            if OCR_DEBUG_EVERY and (frame_count % OCR_DEBUG_EVERY == 0):
                cv2.imwrite(f"debug_ocr_frame_{frame_count:05d}.jpg", thresh_crop)
        else:
            current_time_sec = np.nan

        # --- 2. Core Inference Pipeline ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        augmented = resize_transform(image=frame_rgb)
        resized_rgb = augmented["image"]

        img_tensor = torch.from_numpy(resized_rgb.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.5).cpu().numpy().squeeze().astype(np.uint8)

        final_mask = cv2.resize(pred_mask, (width, height), interpolation=cv2.INTER_NEAREST)

        # --- 3. Geometrical Contour Extraction ---
        processing_mask = final_mask * 255
        kernel = np.ones((7, 7), np.uint8)
        clean_mask = cv2.morphologyEx(processing_mask, cv2.MORPH_OPEN, kernel)

        cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cx, cy, le_x, lowest_y, area_px = None, None, None, None, 0

        if cnts:
            c = max(cnts, key=cv2.contourArea)
            area_px = cv2.contourArea(c)

            if area_px > min_area_px:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                pts = c.reshape(-1, 2)
                xs, ys = pts[:, 0], pts[:, 1]

                # Robust leading edge via percentile (ignores single-pixel spikes).
                if DEPLOYMENT_DIR == "left-to-right":
                    le_x = int(np.percentile(xs, 98))
                elif DEPLOYMENT_DIR == "right-to-left":
                    le_x = int(np.percentile(xs, 2))
                elif DEPLOYMENT_DIR == "bottom-to-top":
                    le_x = int(np.percentile(ys, 2))
                elif DEPLOYMENT_DIR == "top-to-bottom":
                    le_x = int(np.percentile(ys, 98))
                else:
                    le_x = int(np.percentile(xs, 98))

                # Robust LOWEST point (bottom edge) via high percentile of y's.
                # In OpenCV image coords, higher y = lower on screen.
                lowest_y = int(np.percentile(ys, LOWEST_Y_PCTL))

        # Append in lockstep so all arrays stay the same length.
        if le_x is not None and cx is not None and lowest_y is not None:
            raw_leading_edges_px.append(le_x)
            raw_centroids_x.append(cx)
            raw_centroids_y.append(cy)
            raw_lowest_y_px.append(lowest_y)
            raw_areas_px.append(area_px)
            valid_frame_indices.append(frame_count)
            raw_timestamps_sec.append(current_time_sec)

        # --- 4. Dynamic Visual Overlay Rendering ---
        overlay_color = np.zeros_like(frame)
        overlay_color[:] = [0, 0, 255]
        airbag_highlight = cv2.bitwise_and(overlay_color, overlay_color, mask=final_mask)
        blended_frame = cv2.addWeighted(frame, 0.70, airbag_highlight, 0.30, 0)

        # Timer ROI box
        cv2.rectangle(blended_frame, (TIMER_X1, TIMER_Y1), (TIMER_X2, TIMER_Y2), (255, 0, 0), 2)
        if do_ocr and not np.isnan(current_time_sec):
            cv2.putText(blended_frame, f"OCR: {current_time_sec}s",
                        (TIMER_X1, TIMER_Y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Static steering-rim reference line (light blue) — drawn EVERY frame
        cv2.line(blended_frame, (0, STEERING_RIM_Y),
                 (width, STEERING_RIM_Y), (255, 150, 0), 1)
        cv2.putText(blended_frame, "Steering Rim", (10, STEERING_RIM_Y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 0), 1)

        # Per-frame airbag markers
        if cx is not None and cy is not None:
            cv2.circle(blended_frame, (cx, cy), 6, (0, 255, 0), -1)
            # Leading-edge line (yellow, vertical for horizontal deploy)
            if DEPLOYMENT_DIR in ["left-to-right", "right-to-left"]:
                cv2.line(blended_frame, (le_x, 0), (le_x, height), (0, 255, 255), 2)
            else:
                cv2.line(blended_frame, (0, le_x), (width, le_x), (0, 255, 255), 2)

        if lowest_y is not None:
            # Dynamic lowest-point line (orange)
            cv2.line(blended_frame, (0, lowest_y), (width, lowest_y), (0, 165, 255), 1)
            # Real-time drop-below-rim annotation
            current_drop_px = lowest_y - STEERING_RIM_Y
            if current_drop_px > 0:
                drop_mm = current_drop_px * METERS_PER_PIXEL * 1000
                cv2.putText(blended_frame, f"Drop: {drop_mm:.1f} mm",
                            (10, min(lowest_y + 18, height - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        out.write(blended_frame)

cap.release()
out.release()
print(f"Video visualization complete. Exported tracking clip to: {OUTPUT_VIDEO_PATH}")

# ==========================================
# 5. MATHEMATICAL KINEMATICS & SMOOTHING
# ==========================================
if len(valid_frame_indices) < 15:
    print("Error: Too few frames detected containing an airbag silhouette to compute kinematics metrics safely.")
    exit()

# Defensive: every per-frame array must be the same length as the index array.
assert len(raw_leading_edges_px) == len(valid_frame_indices), "leading edge array length mismatch"
assert len(raw_centroids_x)     == len(valid_frame_indices), "centroid X array length mismatch"
assert len(raw_centroids_y)     == len(valid_frame_indices), "centroid Y array length mismatch"
assert len(raw_lowest_y_px)     == len(valid_frame_indices), "lowest-Y array length mismatch"
assert len(raw_areas_px)        == len(valid_frame_indices), "area array length mismatch"
assert len(raw_timestamps_sec)  == len(valid_frame_indices), "OCR time array length mismatch"

valid_frame_indices  = np.array(valid_frame_indices)
raw_leading_edges_px = np.array(raw_leading_edges_px, dtype=float)
raw_lowest_y_px      = np.array(raw_lowest_y_px,      dtype=float)
raw_areas_px         = np.array(raw_areas_px,         dtype=float)

# --- Build a trustworthy time axis from sampled OCR reads ---
raw_times = np.array(raw_timestamps_sec, dtype=float)
valid_mask = ~np.isnan(raw_times)
n_valid = int(np.sum(valid_mask))

use_fallback_fps = False
fit_slope = None

if n_valid >= MIN_OCR_READS:
    frames_v = valid_frame_indices[valid_mask]
    times_v  = raw_times[valid_mask]

    m, b = np.polyfit(frames_v, times_v, 1)
    fitted = m * frames_v + b
    residuals = times_v - fitted
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    print(f"OCR fit (pass 1): {n_valid} reads | slope {m:.6f} s/frame | RMS residual {rms*1000:.2f} ms")

    if rms > 0:
        keep = np.abs(residuals) < OUTLIER_REJECT_SIGMA * rms
        if keep.sum() >= MIN_OCR_READS and keep.sum() < len(frames_v):
            m, b = np.polyfit(frames_v[keep], times_v[keep], 1)
            fitted2 = m * frames_v[keep] + b
            rms = float(np.sqrt(np.mean((times_v[keep] - fitted2) ** 2)))
            print(f"OCR fit (pass 2): kept {int(keep.sum())}/{len(frames_v)} | "
                  f"slope {m:.6f} s/frame | RMS residual {rms*1000:.2f} ms")

    if m <= 0:
        print(f"Warning: OCR fit slope is non-positive ({m:.6f}). Falling back to video FPS.")
        use_fallback_fps = True
    elif rms > MAX_FIT_RMS_SEC:
        print(f"Warning: OCR fit RMS {rms*1000:.1f} ms exceeds {MAX_FIT_RMS_SEC*1000:.0f} ms. "
              f"Falling back to video FPS.")
        use_fallback_fps = True
    else:
        fit_slope = m
        timestamps_sec = m * valid_frame_indices + b
        print(f"OCR-derived effective FPS = {1.0/m:.2f}")
else:
    print(f"Warning: only {n_valid} valid OCR reads (need {MIN_OCR_READS}). "
          f"Falling back to video FPS ({visual_fps:.2f}).")
    use_fallback_fps = True

if use_fallback_fps:
    timestamps_sec = valid_frame_indices / visual_fps

# Zero the time axis at the first detected frame.
timestamps_sec = timestamps_sec - timestamps_sec[0]

# --- Horizontal: displacement / velocity ---
initial_position_px = raw_leading_edges_px[0]
if DEPLOYMENT_DIR in ["right-to-left", "bottom-to-top"]:
    raw_displacement_px = initial_position_px - raw_leading_edges_px
else:
    raw_displacement_px = raw_leading_edges_px - initial_position_px

displacement_meters = raw_displacement_px * METERS_PER_PIXEL
area_sq_meters      = raw_areas_px * (METERS_PER_PIXEL ** 2)

# --- Vertical: drop of lowest point past the steering rim line ---
# Positive = below the rim (further down on screen).
drop_below_rim_px      = raw_lowest_y_px - STEERING_RIM_Y
drop_below_rim_meters  = drop_below_rim_px * METERS_PER_PIXEL

poly_order = 3
n_points   = len(displacement_meters)
window_len = min(11, n_points if n_points % 2 == 1 else n_points - 1)


def safe_savgol(series):
    if window_len < poly_order + 2:
        return np.asarray(series, dtype=float)
    return savgol_filter(series, window_len, poly_order)


smoothed_disp_m  = safe_savgol(displacement_meters)
smoothed_area_m2 = safe_savgol(area_sq_meters)
smoothed_drop_m  = safe_savgol(drop_below_rim_meters)

velocity_mps      = np.gradient(smoothed_disp_m, timestamps_sec)
velocity_smoothed = safe_savgol(velocity_mps)

# ==========================================
# 6. EXPORT COMPILED DATA (CSV)
# ==========================================
df_results = pd.DataFrame({
    'Frame_ID':                valid_frame_indices,
    'True_Video_Time_Sec':     timestamps_sec,
    'Displacement_Meters':     smoothed_disp_m,
    'Velocity_M_per_S':        velocity_smoothed,
    'Drop_Below_Rim_Meters':   smoothed_drop_m,
    'Surface_Area_SqMeters':   smoothed_area_m2,
    'Raw_Centroid_X_Px':       raw_centroids_x,
    'Raw_Centroid_Y_Px':       raw_centroids_y,
    'Raw_Lowest_Y_Px':         raw_lowest_y_px,
})
df_results.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"Physics database log cleanly exported to: {OUTPUT_CSV_PATH}")

peak_v     = float(np.nanmax(velocity_smoothed))
total_disp = float(smoothed_disp_m[-1])
max_drop   = float(np.nanmax(smoothed_drop_m))
print(f"Summary -> peak velocity: {peak_v:.2f} m/s | "
      f"total horizontal displacement: {total_disp*1000:.1f} mm | "
      f"max drop below rim: {max_drop*1000:.1f} mm | "
      f"duration: {timestamps_sec[-1]*1000:.1f} ms"
      + (f" | time source: OCR ({1.0/fit_slope:.1f} fps eq.)" if fit_slope else
         f" | time source: video FPS fallback ({visual_fps:.1f})"))

# ==========================================
# 7. GENERATE ANALYTICAL PLOTS
# ==========================================
fig, axes = plt.subplots(4, 1, figsize=(12, 11))

# (1) Displacement
ax = axes[0]
ax.plot(timestamps_sec, smoothed_disp_m, color='blue', linewidth=2, label="Filtered Tracking Line")
ax.scatter(timestamps_sec, displacement_meters, color='lightskyblue', s=10, alpha=0.5, label="Raw Pixel Inferences")
ax.set_ylabel("Displacement (m)")
ax.set_title("Airbag Deployment Kinematics Analysis Profiles")
ax.grid(True, linestyle='--')
ax.legend(loc="upper left")

# (2) Velocity
ax = axes[1]
ax.plot(timestamps_sec, velocity_smoothed, color='red', linewidth=2)
ax.set_ylabel("Velocity (m/s)")
ax.grid(True, linestyle='--')

# (3) Vertical drop below steering rim
ax = axes[2]
ax.plot(timestamps_sec, smoothed_drop_m, color='orange', linewidth=2)
ax.axhline(0, color='blue', linestyle='--', label="Steering Rim Line")
ax.fill_between(timestamps_sec, smoothed_drop_m, 0,
                where=(smoothed_drop_m > 0), color='orange', alpha=0.3,
                label="Airbag Below Rim")
ax.set_ylabel("Vertical Drop Past Rim (m)")
ax.legend(loc="upper left")
ax.grid(True, linestyle='--')

# (4) Surface area
ax = axes[3]
ax.plot(timestamps_sec, smoothed_area_m2, color='purple', linewidth=2)
ax.set_xlabel("True Elapsed Time (Seconds)")
ax.set_ylabel("Fabric Area (m²)")
ax.grid(True, linestyle='--')

plt.tight_layout()
plt.savefig("airbag_kinematics_plots.png", dpi=300)
print("Analytical charts compiled and saved as: airbag_kinematics_plots.png")
plt.show()
