from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import MDS
import numpy as np
from scipy.spatial import procrustes

app = FastAPI()

model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens-v2")

class WordRequest(BaseModel):
    words: List[str]

# 初期のアンカー位置（原点）に合わせるための基準行列
anchor_template = np.array([[0.0, 0.0]])

@app.post("/position")
async def compute_positions(req: WordRequest):
    words = [w for w in req.words if w is not None and w.strip() != ""]
    if len(words) < 2:
        return {"positions": []}  # アンカー＋最低1語必要

    # BERT埋め込み
    embeddings = model.encode(words)

    # 類似度行列から距離行列
    similarity_matrix = util.cos_sim(embeddings, embeddings).numpy()
    distance_matrix = 1.0 - similarity_matrix

    # MDSによる2次元プロット
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    mds_positions = mds.fit_transform(distance_matrix)

    # プロクラステス変換：アンカー語を原点(0,0)に配置
    anchor_index = 0
    template = np.copy(mds_positions)  # コピーしておく（回転前の位置）
    template[anchor_index] = [0.0, 0.0]  # 原点にするテンプレート

    _, aligned_positions, _ = procrustes(template, mds_positions)

    result = [
        {"word": word, "x": float(pos[0]), "y": float(pos[1])}
        for word, pos in zip(words, aligned_positions)
    ]
    return {"positions": result}
