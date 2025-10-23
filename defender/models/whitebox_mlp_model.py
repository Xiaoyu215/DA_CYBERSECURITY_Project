# whitebox.py (thrember v3, 2568-d)
import json, os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import joblib
from thrember.features import PEFeatureExtractor

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE  # all artifacts live here

WEIGHTS_DEFAULT = MODELS_DIR / "whitebox_mlp.pt"
SCALER_JSON     = MODELS_DIR / "whitebox_scaler_params.json"   # preferred
SCALER_JOBLIB   = MODELS_DIR / "whitebox_scaler.joblib"        # optional fallback
THRESH_PATH     = MODELS_DIR / "whitebox_threshold.json"
META_PATH       = MODELS_DIR / "whitebox_model_meta.json"

STRICT_EXTRACT = os.getenv("STRICT_EXTRACT", "1") == "1"

class WhiteboxMLPEmberModel:
    """
    White-box MLP adapter compatible with the template.
    Uses EMBER2024/thrember PE v3 features at inference.
    """
    def __init__(
        self,
        model_gz_path: str,
        model_thresh: Optional[float] = None,
        model_ball_thresh: Optional[float] = None,
        model_max_history: Optional[int] = None,
        model_name: str = "whitebox_mlp",
    ) -> None:
        self.name = model_name

        # --- Load meta, enforce feature schema ---
        meta = {"feature_version": "ember_v3_pe", "input_dim": 2568, "hidden_dims": [512, 256]}
        if META_PATH.exists():
            meta.update(json.loads(META_PATH.read_text()))
        self.feature_version = str(meta.get("feature_version", "ember_v3_pe"))
        self.input_dim: int  = int(meta.get("input_dim", 2568))
        self.hidden_dims     = list(meta.get("hidden_dims", [512, 256]))

        if self.feature_version != "ember_v3_pe":
            raise RuntimeError(f"Feature version mismatch: expected 'ember_v3_pe', got '{self.feature_version}'")

        # --- Threshold (val@1%FPR recommended) ---
        self.threshold = float(model_thresh) if model_thresh is not None else 0.5
        if THRESH_PATH.exists():
            try:
                obj = json.loads(THRESH_PATH.read_text())
                self.threshold = float(obj.get("threshold", self.threshold))
            except Exception as e:
                raise RuntimeError(f"[whitebox] failed to read threshold {THRESH_PATH}: {e}")

        # --- Scaler: prefer JSON mean/scale (version-proof); fallback to joblib if present ---
        self.scaler_mean = None
        self.scaler_scale = None
        if SCALER_JSON.exists():
            ps = json.loads(SCALER_JSON.read_text())
            self.scaler_mean  = np.asarray(ps["mean"],  dtype=np.float32)
            self.scaler_scale = np.asarray(ps["scale"], dtype=np.float32)
            if self.scaler_mean.size != self.input_dim or self.scaler_scale.size != self.input_dim:
                raise RuntimeError("Scaler JSON size mismatch with input_dim")
            self.scaler = None  # use arrays
        elif SCALER_JOBLIB.exists():
            self.scaler = joblib.load(SCALER_JOBLIB)
            n_in = getattr(self.scaler, "n_features_in_", len(getattr(self.scaler, "mean_", [])))
            if n_in != self.input_dim:
                raise RuntimeError(f"Scaler expects {n_in} features but meta says {self.input_dim}")
        else:
            raise RuntimeError("No scaler found (need whitebox_scaler_params.json or whitebox_scaler.joblib)")

        # --- Build MLP and load weights ---
        import torch, torch.nn as nn
        class MLP(nn.Module):
            def __init__(self, d: int, hs):
                super().__init__()
                layers, last = [], d
                for h in hs:
                    layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(0.10)]
                    last = h
                layers += [nn.Linear(last, 1)]
                self.net = nn.Sequential(*layers)
            def forward(self, x): return self.net(x).squeeze(-1)

        self.torch = torch
        self.model = MLP(self.input_dim, self.hidden_dims)

        weights_path = WEIGHTS_DEFAULT
        self._load_checkpoint(weights_path)
        self.model.eval()

        # --- Feature extractor: thrember (EMBER2024 v3 PE) ---
        try:
            import thrember
            self._thrember = thrember
            self._extract_ok = True
            self.extractor = PEFeatureExtractor()
        except Exception as e:
            self._thrember = None
            self._extract_ok = False
            if STRICT_EXTRACT:
                raise RuntimeError(f"thrember import failed: {e}")

    # ----- Public API -----
    def predict(self, bytez: bytes) -> int:

        v = self._bytes_to_features(bytez)  
        print(v)
        x = self._scale(v.reshape(1, -1))   

        # torch inference
        with self.torch.no_grad():
            t = self.torch.from_numpy(x.astype(np.float32))
            logits = self.model(t)
            if not self.torch.isfinite(logits).all():
                raise RuntimeError("Non-finite logits from model")
            prob = float(self.torch.sigmoid(logits).cpu().numpy().ravel()[0])

        if not (0.0 <= prob <= 1.0) or not np.isfinite(prob):
            raise RuntimeError(f"Invalid probability {prob}")

        return int(prob >= self.threshold)

    def model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "feature_version": self.feature_version,
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "threshold": float(self.threshold),
            "has_scaler_json": bool(self.scaler_mean is not None),
            "has_scaler_joblib": bool(SCALER_JOBLIB.exists()),
            "extractor": "thrember",
            "extractor_ok": bool(self._extract_ok),
        }

    # ----- Internals -----
    def _scale(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_mean is not None:
            return (X - self.scaler_mean) / self.scaler_scale
        else:
            return self.scaler.transform(X).astype(np.float32)
    

    def _load_checkpoint(self, path: Path) -> None:
        ckpt = self.torch.load(path, map_location="cpu")

        def strip_module(sd):
            return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = strip_module(ckpt["state_dict"])
            try: self.model.load_state_dict(sd, strict=True)
            except Exception: self.model.load_state_dict(sd, strict=False)
        elif isinstance(ckpt, dict):
            sd = strip_module(ckpt)
            try: self.model.load_state_dict(sd, strict=True)
            except Exception: self.model.load_state_dict(sd, strict=False)
        elif isinstance(ckpt, self.torch.jit.ScriptModule):
            self.model = ckpt
        else:
            self.model = ckpt  # full module

    def _bytes_to_features(self, bytez: bytes) -> np.ndarray:
        """
        Raw PE bytes -> EMBER v3 (2568-d) feature vector using Thrember.
        """
        if self._extract_ok:
            try:
                vec = self.extractor.feature_vector(bytez)
                v = np.asarray(vec, dtype=np.float32)

                # Defensive pad/trim to expected dim
                if v.shape[0] != self.input_dim:
                    vv = np.zeros((self.input_dim,), dtype=np.float32)
                    n = min(self.input_dim, v.shape[0])
                    vv[:n] = v[:n]
                    v = vv
                return v
            except Exception as e:
                if STRICT_EXTRACT:
                    # Force HTTP 500 via the Flask app — expected in strict mode
                    raise RuntimeError("feature extraction failed") from e

        # Fallback (STRICT_EXTRACT=0): zero vector to keep API responsive
        return np.zeros((self.input_dim,), dtype=np.float32)


    def _pad_or_trim(self, v: np.ndarray) -> np.ndarray:
        if v.shape[0] == self.input_dim:
            return v
        out = np.zeros((self.input_dim,), dtype=np.float32)
        n = min(self.input_dim, v.shape[0]); out[:n] = v[:n]
        return out
