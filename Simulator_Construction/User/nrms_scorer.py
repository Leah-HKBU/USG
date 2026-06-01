"""
NRMS Scorer

Provides NRMSScorer — a fast inference interface for the trained NRMS model.

"""

import os
import sys
import pickle
from typing import List

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules import AttentionPooling

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoint')
DEVICE   = torch.device('cpu')   # scoring on CPU (fast enough for inference)


class _UserEncoder(nn.Module):
    """Light-weight user encoder: attention pooling over pre-computed news vecs."""
    def __init__(self, d=64):
        super().__init__()
        self.attn_pool = AttentionPooling(d, d)

    def forward(self, x):
        # x: [1, MAX_CLICK, 64]
        return self.attn_pool(x)   # [1, 64]


class NRMSScorer:
    """Fast NRMS collaborative signal scorer using pre-computed embeddings.

    Loads news_scoring.npy (all news embeddings) and the saved user encoder
    attention weights. Scoring = look-up + attention pool + sigmoid dot product.

    Accepts string news IDs ('N10000') and converts via news_index.
    """

    _instance: 'NRMSScorer' = None

    @classmethod
    def get_instance(cls, checkpoint_path: str = None) -> 'NRMSScorer':
        """Return shared singleton instance (model loaded once)."""
        if cls._instance is None:
            cls._instance = cls(checkpoint_path)
        return cls._instance

    def __init__(self, checkpoint_path: str = None):
        ckpt = checkpoint_path or os.path.join(CKPT_DIR, 'nrms_best.pt')

        missing = [p for p in [ckpt,
                                os.path.join(CKPT_DIR, 'news_scoring.npy'),
                                os.path.join(CKPT_DIR, 'news_index.pkl')]
                   if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"Missing files: {missing}\n"
                "Run preprocess.py then train_nrms.py first."
            )

        # Pre-computed news embeddings: [n_news+1, 64]
        self.news_scoring = np.load(os.path.join(CKPT_DIR, 'news_scoring.npy'))

        # String → int mapping
        with open(os.path.join(CKPT_DIR, 'news_index.pkl'), 'rb') as f:
            self.news_index = pickle.load(f)

        with open(os.path.join(CKPT_DIR, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        self.MAX_CLICK_LEN = meta.get('MAX_CLICK_LEN', 50)

        # Load only the user-encoder attention pooling weights from checkpoint
        self.user_encoder = _UserEncoder(d=64).to(DEVICE)
        state = torch.load(ckpt, map_location=DEVICE)
        # Extract attn_pool_news weights (stored as "attn_pool_news.*" in NRMS)
        pool_state = {k.replace('attn_pool_news.', ''): v
                      for k, v in state.items()
                      if k.startswith('attn_pool_news.')}
        self.user_encoder.attn_pool.load_state_dict(pool_state)
        self.user_encoder.eval()

        print(f"[NRMSScorer] Loaded — {len(self.news_index)} news items, "
              f"scoring shape: {self.news_scoring.shape}")

    def _to_int_ids(self, str_ids: List[str], max_len: int = None) -> List[int]:
        """Convert string news IDs to integer indices, filtering unknowns."""
        int_ids = [self.news_index[s] for s in str_ids if s in self.news_index]
        if max_len:
            int_ids = int_ids[-max_len:]
        return int_ids

    def _user_vec(self, click_str_ids: List[str]) -> np.ndarray:
        """Compute 64-dim user vector from click history via attention pooling."""
        int_ids = self._to_int_ids(click_str_ids, self.MAX_CLICK_LEN)
        if not int_ids:
            return np.zeros(64, dtype=np.float32)

        # Pad left with 0 to MAX_CLICK_LEN
        padded = [0] * (self.MAX_CLICK_LEN - len(int_ids)) + int_ids
        vecs = self.news_scoring[padded]                        # [MAX_CLICK, 64]
        x    = torch.FloatTensor(vecs).unsqueeze(0).to(DEVICE) # [1, MAX_CLICK, 64]
        with torch.no_grad():
            uv = self.user_encoder(x)                          # [1, 64]
        return uv.squeeze(0).cpu().numpy()                     # [64]

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def get_score(self, click_str_ids: List[str], candidate_str_id: str) -> float:
        """Compute S = sigmoid(user_vec · cand_vec).

        Args:
            click_str_ids:    user's clicked news IDs as strings, e.g. ['N41340', ...]
            candidate_str_id: candidate news ID, e.g. 'N58377'

        Returns:
            float in [0, 1]:  ≥0.6 = high behavioral relevance, <0.4 = low
            Returns 0.5 if candidate not in index.
        """
        cand_int = self.news_index.get(candidate_str_id, 0)
        if cand_int == 0:
            return 0.5

        user_vec = self._user_vec(click_str_ids)               # [64]
        cand_vec = self.news_scoring[cand_int]                 # [64]
        return self._sigmoid(float(np.dot(user_vec, cand_vec)))

    def batch_score(self, click_str_ids: List[str],
                    candidate_str_ids: List[str]) -> List[float]:
        """Score multiple candidates for the same user in one pass.

        More efficient than looping get_score() when evaluating many candidates.
        """
        user_vec = self._user_vec(click_str_ids)   # [64]

        scores = []
        for cid in candidate_str_ids:
            int_id = self.news_index.get(cid, 0)
            if int_id == 0:
                scores.append(0.5)
            else:
                raw = float(np.dot(user_vec, self.news_scoring[int_id]))
                scores.append(self._sigmoid(raw))
        return scores
