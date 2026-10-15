"""Pretrained ReDimNet-M embeddings for ALL train utterances (full length) -> exp/emb cache."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sred.data import TRAIN_DIR, read_train_list  # noqa: E402
from sred.embed import extract_cached  # noqa: E402
from sred.models import load_model  # noqa: E402

items = read_train_list()
emb = extract_cached("redimnet_M_mix", lambda: load_model("redimnet_M_mix"), "trainall",
                     [r for _, r in items], TRAIN_DIR, mode="full")
print("done", emb.shape)
