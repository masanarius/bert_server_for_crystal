from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import MDS
import numpy as np

app = FastAPI()

# 日本語BERTモデルの読み込み
model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens-v2")

# リクエストの形式定義
class WordRequest(BaseModel):
    words: List[str]

@app.post("/position")
async def compute_positions(req: WordRequest):
    # 空文字やNoneを除外
    words = [w for w in req.words if w is not None and w.strip() != ""]
    if len(words) < 2:
        return {"positions": []}  # アンカー＋1語以上が必要

    # 文ベクトル化
    embeddings = model.encode(words)

    # 類似度行列から距離行列へ
    similarity_matrix = util.cos_sim(embeddings, embeddings).numpy()
    distance_matrix = 1.0 - similarity_matrix

    # MDSで2次元座標に変換
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    mds_positions = mds.fit_transform(distance_matrix)

    # アンカー語を原点に平行移動
    anchor_pos = mds_positions[0]
    aligned_positions = mds_positions - anchor_pos

    # 結果の整形と返却
    result = [
        {"word": word, "x": float(pos[0]), "y": float(pos[1])}
        for word, pos in zip(words, aligned_positions)
    ]
    return {"positions": result}
