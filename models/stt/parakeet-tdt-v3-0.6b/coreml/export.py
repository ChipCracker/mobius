# file: models/stt/parakeet-tdt-v3-0.6b/coreml/export.py
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

class JointWrapper(nn.Module):
    """
    Wrappt das bestehende Joint-Modul und liefert zusätzlich zu den
    Argmax-Entscheidungen auch die rohen Logits für Token- und Dauer-Head.
    """
    def __init__(self, joint: nn.Module):
        super().__init__()
        self.joint = joint

    @torch.no_grad()
    def forward(self, encoder_step: torch.Tensor, decoder_step: torch.Tensor):
        """
        Erwartete Eingaben:
          encoder_step: (1, H_enc, 1) float32
          decoder_step: (1, H_dec, 1) float32
        Rückgaben:
          - token_logits: (1, V, 1) float32
          - duration_logits: (1, D, 1) float32
          - token_id: (1, 1, 1) int32
          - token_prob: (1, 1, 1) float32  (max. Token-Prob)
          - duration: (1, 1, 1) int32
        """
        out = self.joint(encoder_step, decoder_step)

        # Unterstütze beide Varianten: tuple oder dict aus dem bestehenden Joint
        if isinstance(out, dict):
            token_logits = out["token_logits"]                 # (1, V, 1) oder (1, V)
            duration_logits = out["duration_logits"]           # (1, D, 1) oder (1, D)
        else:
            # Konvention: joint(...) -> (token_logits, duration_logits)
            token_logits, duration_logits = out

        # Normiere evtl. 2D-Logits auf 3D (B, C, 1), damit es exakt zu Swift passt
        if token_logits.dim() == 2:
            token_logits = token_logits.unsqueeze(-1)
        if duration_logits.dim() == 2:
            duration_logits = duration_logits.unsqueeze(-1)

        # Token-Entscheidung + max. Wahrscheinlichkeit (Softmax über Kanalachse)
        token_probs = torch.softmax(token_logits, dim=1)                      # (1, V, 1)
        token_id = torch.argmax(token_probs, dim=1, keepdim=True).to(torch.int32)  # (1, 1, 1)
        # Gather über Kanalachse 1
        token_prob = torch.gather(token_probs, 1, token_id).to(torch.float32)      # (1, 1, 1)

        # Dauer-Entscheidung per Argmax über Kanalachse
        duration = torch.argmax(duration_logits, dim=1, keepdim=True).to(torch.int32)  # (1, 1, 1)

        return {
            "token_logits": token_logits.to(torch.float32),
            "duration_logits": duration_logits.to(torch.float32),
            "token_id": token_id,
            "token_prob": token_prob,
            "duration": duration,
        }
