# whitebox.py
"""
White-box MLP model adapter for the older HTTP template.
- Exposes WhiteboxMLPEmberModel with predict(bytez)->0/1 and model_info().
- Loads: whitebox_mlp.pt, whitebox_scaler.joblib, whitebox_threshold.json, whitebox_model_meta.json
- Extracts EMBER-style (2381-d) features from raw PE bytes via ember + LIEF.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import joblib

# ----------------------- Config / Paths -----------------------
HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE

WEIGHTS_DEFAULT = MODELS_DIR / "whitebox_mlp.pt"
SCALER_PATH     = MODELS_DIR / "whitebox_scaler.joblib"
THRESH_PATH     = MODELS_DIR / "whitebox_threshold.json"
META_PATH       = MODELS_DIR / "whitebox_model_meta.json"

# Extraction policy:
# If STRICT_EXTRACT=1, raise on extraction failure (HTTP 500).
# Else, fall back to a zero vector (still returns a 0/1 decision).
STRICT_EXTRACT = os.getenv("STRICT_EXTRACT", "0") == "1"

# ----------------------- Model Adapter -----------------------
class WhiteboxMLPEmberModel:
    """
    Constructor signature mirrors the template's Ember model so main.py changes are minimal.
    Only model_gz_path and model_thresh are actually used.
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

        # ---- Threshold
        self.threshold = float(model_thresh) if model_thresh is not None else 0.5
        try:
            obj = json.loads(THRESH_PATH.read_text())
            self.threshold = float(obj.get("threshold", self.threshold))
            print(f"[whitebox] Loaded threshold from {THRESH_PATH}: {self.threshold}")
        except Exception as e:
            raise RuntimeError(f"[whitebox] threshold reading error {THRESH_PATH}: {e}")

        # ---- Scaler
        try:
            self.scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            raise RuntimeError(f"[whitebox] scaler reading error {SCALER_PATH}: {e}")
        # ---- Meta (must match training architecture)
        meta = {"input_dim": 2381, "hidden_dims": [512, 256]}
        try:
            meta.update(json.loads(META_PATH.read_text()))
        except Exception as e:
            raise RuntimeError(f"[whitebox] meta data reading error {META_PATH}: {e}")
        self.input_dim: int = int(meta["input_dim"])
        self.hidden_dims = list(meta["hidden_dims"])

        # Optional sanity check: scaler feature count
        if self.scaler is not None:
            n_in = getattr(self.scaler, "n_features_in_", len(getattr(self.scaler, "mean_", [])))
            if n_in != self.input_dim:
                raise RuntimeError(f"Scaler expects {n_in} features but meta says {self.input_dim}")

        # ---- Build MLP and load checkpoint
        import torch
        import torch.nn as nn

        class MLP(nn.Module):
            def __init__(self, d: int, hs):
                super().__init__()
                layers, last = [], d
                for h in hs:
                    layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(0.10)]
                    last = h
                layers += [nn.Linear(last, 1)]
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x).squeeze(-1)

        self.torch = torch
        self.model = MLP(self.input_dim, self.hidden_dims)

        weights_path = Path(model_gz_path)
        if not weights_path.exists():
            weights_path = WEIGHTS_DEFAULT
        self._load_checkpoint(weights_path)

        self.model.eval()

        # ---- Feature extractor: EMBER (uses LIEF)
        self._ember_ok = False
        self.extractor = None
        try:
            import ember  # noqa: F401
            # Delay import to here to avoid import errors at module import time
            import ember as _ember
            self.extractor = _ember.PEFeatureExtractor()
            self._ember_ok = True
        except Exception:
            self._ember_ok = False  # You can vendor the extractor if needed

    # ----------------------- Public API -----------------------
    def predict(self, bytez: bytes) -> int:
        """
        Bytes in -> 0/1 out (0=benign, 1=malicious)
        """
        x = self._bytes_to_features(bytez).reshape(1, -1)
        print("features",x)
        if self.scaler is not None:
            x = self.scaler.transform(x).astype(np.float32)

        with self.torch.no_grad():
            t = self.torch.from_numpy(x.astype(np.float32))
            prob = float(self.torch.sigmoid(self.model(t)).cpu().numpy().ravel()[0])
        print(prob, self.threshold)
        return int(prob >= self.threshold)

    def model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "threshold": float(self.threshold),
            "has_scaler": bool(self.scaler is not None),
            "ember_extractor": bool(self._ember_ok),
        }

    # ----------------------- Internals -----------------------
    def _load_checkpoint(self, path: Path) -> None:
        """
        Robustly load:
          - pure state_dict
          - {"state_dict": ...}
          - full nn.Module saved with torch.save(model, ...)
          - TorchScript module
        Also strips a leading 'module.' (DataParallel) prefix if present.
        """
        ckpt = self.torch.load(path, map_location="cpu")

        def _strip_module(sd):
            return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

        loaded = False
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = _strip_module(ckpt["state_dict"])
            try:
                self.model.load_state_dict(sd, strict=True); loaded = True
            except Exception:
                self.model.load_state_dict(sd, strict=False); loaded = True
        elif isinstance(ckpt, dict):
            sd = _strip_module(ckpt)
            try:
                self.model.load_state_dict(sd, strict=True); loaded = True
            except Exception:
                self.model.load_state_dict(sd, strict=False); loaded = True
        elif isinstance(ckpt, self.torch.jit.ScriptModule):
            self.model = ckpt; loaded = True
        else:
            # Full nn.Module
            self.model = ckpt; loaded = True

        if not loaded:
            raise RuntimeError("Could not load weights: incompatible state_dict/module")

    def _bytes_to_features(self, bytez: bytes) -> np.ndarray:
        """
        Raw PE bytes -> EMBER-style vector (2381-d by default).
        """
        if self._ember_ok and self.extractor is not None:
            try:
                if hasattr(self.extractor, "feature_vector"):
                    vec = self.extractor.feature_vector(bytez)
                else:
                    vec = self.extractor.extract(bytez)
                v = np.asarray(vec, dtype=np.float32)
                print("Feature vector:",v)
                # Defensive pad/trim
                if v.shape[0] != self.input_dim:
                    print("ERROR dim mismatch", v.shape, self.input_dim)
                    vv = np.zeros((self.input_dim,), dtype=np.float32)
                    n = min(self.input_dim, v.shape[0]); vv[:n] = v[:n]
                    v = vv
                return v
            except Exception as e:
                if STRICT_EXTRACT:
                    raise RuntimeError(f"feature extraction failed: {e}")

        # Fallback: zero vector (keeps API responsive)
        return np.zeros((self.input_dim,), dtype=np.float32)