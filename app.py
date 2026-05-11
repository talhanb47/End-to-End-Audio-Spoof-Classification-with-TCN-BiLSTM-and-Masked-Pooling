#!/usr/bin/env python3
import os
import time
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

BASE_DIR = Path(__file__).resolve().parent

import numpy as np
import librosa

import torch
import torch.nn as nn

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates


# =======================
# USER SETTINGS
# =======================
SR = 16000
N_MFCC = 13
N_FFT = 1024
HOP = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_PATH = os.getenv(
    "CKPT_PATH",
    str(BASE_DIR / "models" / "best_AN_asvspoof2019_LA_fullclip_byEER.pth"),
)
THR_P_BONAFIDE = 0.50

MAX_WAVE_POINTS = 6000
MAX_MFCC_FRAMES = 600


# =======================
# MODEL
# =======================
class AudioNetFullClip(nn.Module):
    def __init__(self, in_dim=13, tcn_hidden=64, lstm_h=128, emb_dim=256, drop=0.4):
        super().__init__()

        self.tcn = nn.Sequential(
            nn.Conv1d(in_dim, tcn_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(tcn_hidden, tcn_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(drop),
        )

        self.blstm = nn.LSTM(
            tcn_hidden,
            lstm_h,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.emb = nn.Sequential(
            nn.Linear(lstm_h * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(drop),
        )

        self.cls = nn.Linear(emb_dim, 2)

    def forward(self, x, lengths):
        # x: (B, T, D)
        B, T, D = x.shape

        z = x.permute(0, 2, 1)  # (B, D, T)
        z = self.tcn(z)         # (B, hidden, T)
        z = z.permute(0, 2, 1)  # (B, T, hidden)

        lengths_cpu = lengths.detach().cpu()

        packed = nn.utils.rnn.pack_padded_sequence(
            z,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, _ = self.blstm(packed)

        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=T,
        )

        mask = (
            torch.arange(T, device=x.device)[None, :] < lengths[:, None]
        ).float()

        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        pooled = (lstm_out * mask.unsqueeze(-1)).sum(dim=1) / denom

        feat256 = self.emb(pooled)
        logits = self.cls(feat256)

        return logits, feat256


# =======================
# HELPERS
# =======================
def downsample_1d(x: np.ndarray, max_points: int) -> np.ndarray:
    if x.size <= max_points:
        return x

    idx = np.linspace(0, x.size - 1, max_points).astype(int)
    return x[idx]


def downsample_mfcc(mfcc: np.ndarray, max_frames: int) -> np.ndarray:
    T = mfcc.shape[1]

    if T <= max_frames:
        return mfcc

    idx = np.linspace(0, T - 1, max_frames).astype(int)
    return mfcc[:, idx]


def elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 3)


def cuda_sync_if_needed():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def get_gpu_memory_mb():
    if DEVICE != "cuda":
        return None

    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 3),
        "reserved_mb": round(torch.cuda.memory_reserved() / (1024 ** 2), 3),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 3),
    }


def performance_label(real_time_factor):
    if real_time_factor is None:
        return "Unknown"

    if real_time_factor < 0.25:
        return "Very fast"

    if real_time_factor < 1.0:
        return "Real-time capable"

    return "Slower than real time"


# =======================
# FASTAPI APP
# =======================
MODEL = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL

    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    model = AudioNetFullClip(in_dim=N_MFCC).to(DEVICE)

    sd = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()

    MODEL = model

    print("=" * 70)
    print("✅ Model loaded successfully")
    print(f"✅ Device: {DEVICE}")
    print(f"✅ Checkpoint: {CKPT_PATH}")
    print("=" * 70)

    yield

    MODEL = None

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("✅ Model unloaded")


app = FastAPI(
    title="End-to-End Audio Spoof Classification",
    description="TCN–BiLSTM and Masked Pooling lightweight web application",
    version="1.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# =======================
# ROUTES
# =======================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/ping")
def ping():
    return {
        "ok": True,
        "device": DEVICE,
        "ckpt": CKPT_PATH,
        "model_loaded": MODEL is not None,
        "sr": SR,
        "n_mfcc": N_MFCC,
        "architecture": "TCN-BiLSTM-MaskedPooling",
    }


@app.post("/api/infer")
async def infer(file: UploadFile = File(...)):
    request_t0 = time.perf_counter()
    timings = {}

    if MODEL is None:
        return JSONResponse(
            {"error": "Model not loaded"},
            status_code=500,
        )

    # -----------------------
    # Read uploaded audio
    # -----------------------
    t0 = time.perf_counter()
    data = await file.read()
    timings["upload_read_ms"] = elapsed_ms(t0)

    if not data:
        return JSONResponse(
            {"error": "Empty upload"},
            status_code=400,
        )

    filename = file.filename or "uploaded_audio"
    suffix = os.path.splitext(filename)[1].lower()
    file_size_bytes = len(data)

    # -----------------------
    # Temporary file write
    # -----------------------
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    timings["temp_write_ms"] = elapsed_ms(t0)

    # -----------------------
    # Audio decoding
    # -----------------------
    try:
        t0 = time.perf_counter()
        y, _ = librosa.load(tmp_path, sr=SR, mono=True)
        timings["audio_decode_ms"] = elapsed_ms(t0)

    except Exception as e:
        return JSONResponse(
            {"error": f"Audio decode failed: {repr(e)}"},
            status_code=400,
        )

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    if y is None or len(y) == 0:
        return JSONResponse(
            {"error": "Decoded audio is empty"},
            status_code=400,
        )

    duration_s = float(len(y) / SR)

    # -----------------------
    # MFCC extraction
    # -----------------------
    try:
        t0 = time.perf_counter()
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=SR,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP,
        )
        timings["mfcc_extraction_ms"] = elapsed_ms(t0)

    except Exception as e:
        return JSONResponse(
            {"error": f"MFCC computation failed: {repr(e)}"},
            status_code=500,
        )

    T = int(mfcc.shape[1])

    # -----------------------
    # Tensor preparation
    # -----------------------
    t0 = time.perf_counter()

    x = torch.tensor(
        mfcc.T,
        dtype=torch.float32,
    ).unsqueeze(0).to(DEVICE)

    lengths = torch.tensor(
        [x.shape[1]],
        dtype=torch.long,
    ).to(DEVICE)

    timings["tensor_preparation_ms"] = elapsed_ms(t0)

    # -----------------------
    # Model inference
    # -----------------------
    try:
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()

        cuda_sync_if_needed()
        t0 = time.perf_counter()

        with torch.no_grad():
            logits, emb = MODEL(x, lengths)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        cuda_sync_if_needed()
        timings["model_forward_ms"] = elapsed_ms(t0)

    except Exception as e:
        return JSONResponse(
            {"error": f"Model forward failed: {repr(e)}"},
            status_code=500,
        )

    # -----------------------
    # Post-processing
    # -----------------------
    t0 = time.perf_counter()

    p_bona = float(probs[0])
    p_spoof = float(probs[1])

    pred = "REAL" if p_bona >= THR_P_BONAFIDE else "FAKE"
    is_spoof = pred == "FAKE"

    confidence = max(p_bona, p_spoof)
    probability_margin = abs(p_bona - p_spoof)

    y_plot = downsample_1d(
        y.astype(np.float32),
        MAX_WAVE_POINTS,
    )

    t_plot = np.linspace(
        0,
        duration_s,
        num=y_plot.size,
    ).astype(np.float32)

    mfcc_plot = downsample_mfcc(
        mfcc.astype(np.float32),
        MAX_MFCC_FRAMES,
    )

    timings["postprocess_payload_ms"] = elapsed_ms(t0)
    timings["server_total_ms"] = elapsed_ms(request_t0)

    server_total_s = timings["server_total_ms"] / 1000.0

    real_time_factor = (
        server_total_s / duration_s
        if duration_s > 0
        else None
    )

    frames_per_second = (
        T / server_total_s
        if server_total_s > 0
        else None
    )

    label = performance_label(real_time_factor)

    metrics = {
        "device": DEVICE,
        "timings": timings,

        "server_total_ms": timings["server_total_ms"],
        "upload_read_ms": timings["upload_read_ms"],
        "temp_write_ms": timings["temp_write_ms"],
        "audio_decode_ms": timings["audio_decode_ms"],
        "mfcc_extraction_ms": timings["mfcc_extraction_ms"],
        "tensor_preparation_ms": timings["tensor_preparation_ms"],
        "model_forward_ms": timings["model_forward_ms"],
        "postprocess_payload_ms": timings["postprocess_payload_ms"],

        "real_time_factor": real_time_factor,
        "performance_label": label,
        "real_time_capable": bool(real_time_factor is not None and real_time_factor < 1.0),

        "frames_per_second": frames_per_second,
        "file_size_bytes": file_size_bytes,
        "duration_s": duration_s,
        "mfcc_frames": T,

        "confidence": float(confidence),
        "probability_margin": float(probability_margin),

        "gpu_memory": get_gpu_memory_mb(),
    }

    report = {
        "title": "End-to-End Audio Spoof Classification Evaluation Report",
        "architecture": "TCN-BiLSTM with Masked Mean Pooling",
        "feature_frontend": "MFCC",
        "sample_rate": SR,
        "prediction": pred,
        "confidence": float(confidence),
        "probability_margin": float(probability_margin),
        "server_total_ms": timings["server_total_ms"],
        "model_forward_ms": timings["model_forward_ms"],
        "real_time_factor": real_time_factor,
        "performance_label": label,
        "device": DEVICE,
        "summary": (
            f"The uploaded audio was classified as {pred}. "
            f"The model confidence was {confidence:.4f}, with a probability margin of "
            f"{probability_margin:.4f}. The backend completed processing in "
            f"{timings['server_total_ms']:.2f} ms, including "
            f"{timings['model_forward_ms']:.2f} ms for the TCN-BiLSTM forward pass. "
            f"The real-time factor was {real_time_factor:.4f}, indicating: {label}."
            if real_time_factor is not None
            else
            f"The uploaded audio was classified as {pred}. "
            f"The model confidence was {confidence:.4f}."
        ),
    }

    return {
        "filename": filename,
        "sr": SR,
        "duration_s": duration_s,
        "file_size_bytes": file_size_bytes,

        "mfcc_frames": T,
        "threshold": float(THR_P_BONAFIDE),

        "p_bonafide": p_bona,
        "p_spoof": p_spoof,
        "confidence": float(confidence),
        "probability_margin": float(probability_margin),

        "pred": pred,
        "is_spoof": is_spoof,

        "wave_t": t_plot.tolist(),
        "wave_y": y_plot.tolist(),
        "mfcc": mfcc_plot.tolist(),
        "emb_shape": list(emb.detach().cpu().numpy().shape),

        "metrics": metrics,
        "report": report,

        "pipeline": {
            "audio_decode": "completed",
            "mfcc_extraction": "completed",
            "tensor_preparation": "completed",
            "tcn_bilstm_forward": "completed",
            "masked_pooling": "completed",
            "classification": "completed",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )