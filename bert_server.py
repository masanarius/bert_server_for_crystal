from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import MDS
import numpy as np

app = FastAPI()

# 日本語BERTモデル
model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens")

# 前回の語を保存
previous_words: List[str] = []

# リクエストの形式
class WordRequest(BaseModel):
    words: List[str]

@app.post("/position")
async def compute_positions(req: WordRequest):
    global previous_words

    # 空やnullな語を除外
    current_words = [w for w in req.words if w is not None and w.strip() != ""]
    previous_valid = [w for w in previous_words if w is not None and w.strip() != ""]

    all_words = previous_valid + current_words
    if len(current_words) == 0:
        return {"positions": []}  # 返す語がない場合

    # BERT埋め込み
    embeddings = model.encode(all_words)

    # 類似度 → 距離行列
    similarity_matrix = util.cos_sim(embeddings, embeddings).numpy()
    distance_matrix = 1.0 - similarity_matrix

    # MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    positions = mds.fit_transform(distance_matrix)

    # 今回の語のインデックス部分のみ返す
    current_positions = positions[-len(current_words):]

    # 有効な語のみ保存
    previous_words = current_words

    result = [
        {"word": word, "x": float(pos[0]), "y": float(pos[1])}
        for word, pos in zip(current_words, current_positions)
    ]
    return {"positions": result}
