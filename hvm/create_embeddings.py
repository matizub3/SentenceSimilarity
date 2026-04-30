import os
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Octen/Octen-Embedding-8B"
BATCH_SIZE = 4

LANG_PAIRS = [
    "ar-ar",
    "en-ar",
    "en-de",
    "en-en",
    "en-tr",
    "es-en",
    "es-es",
    "fr-en",
    "it-en",
    "nl-en",
    "ko-ko",
]

# Save output in the same directory as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(SCRIPT_DIR, "sts17_octen_features.npz")

sentences1 = []
sentences2 = []
scores = []
pairs = []

for pair in LANG_PAIRS:
    print(f"Loading {pair}")

    dataset_dict = load_dataset("mteb/sts17-crosslingual-sts", pair)
    split_name = "test" if "test" in dataset_dict else list(dataset_dict.keys())[0]
    dataset = dataset_dict[split_name]

    sentences1.extend(dataset["sentence1"])
    sentences2.extend(dataset["sentence2"])
    scores.extend(dataset["score"])
    pairs.extend([pair] * len(dataset))

sentence1_N = np.array(sentences1, dtype=object)
sentence2_N = np.array(sentences2, dtype=object)
y_raw_N = np.array(scores, dtype=np.float32)
pair_N = np.array(pairs, dtype=object)

print("Total examples:", len(y_raw_N))

model = SentenceTransformer(
    MODEL_NAME,
    device="cuda",
    trust_remote_code=True,
    model_kwargs={"torch_dtype": torch.bfloat16},
)

print("Encoding sentence1...")
e1_NE = model.encode(
    sentence1_N.tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=False,
).astype(np.float32)

print("Encoding sentence2...")
e2_NE = model.encode(
    sentence2_N.tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=False,
).astype(np.float32)

x_ND = np.concatenate(
    [e1_NE, e2_NE, np.abs(e1_NE - e2_NE)],
    axis=1,
).astype(np.float32)

np.savez_compressed(
    OUT_FILE,
    x_ND=x_ND,
    y_raw_N=y_raw_N,
    pair_N=pair_N,
    sentence1_N=sentence1_N,
    sentence2_N=sentence2_N,
    embedding_model=np.array(MODEL_NAME),
)

print("Saved:", OUT_FILE)
print("x_ND shape:", x_ND.shape)
print("y_raw_N shape:", y_raw_N.shape)
print("pair_N shape:", pair_N.shape)