# -*- coding: utf-8 -*-
"""
01-03➡벡터— 방향과 크기를 가진 화살표

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 01권  수포자들을 위한 머신러닝 수학 완전기초와 파이썬 맛보기/01-03➡벡터— 방향과 크기를 가진 화살표.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 1단계 🟢⭐ "북쪽으로 3km" vs "그냥 3km"
import numpy as np

distance = 3                        # 스칼라: 크기만
move = np.array([0, 3])             # 벡터: 크기 + 방향

print("스칼라:", distance, "| 타입:", type(distance).__name__, "| shape: 없음")
print("벡터  :", move, "| 차원:", move.ndim, "| shape:", move.shape)
print("벡터의 크기:", np.linalg.norm(move))


# %% [Block 2] 6단계 🔵⭐ 내적 — "우리 같은 쪽 보고 있나요?"를 숫자로
import numpy as np

a = np.array([1, 2])          # 벡터 a
b = np.array([3, 4])          # 벡터 b

print("덧셈      :", a + b)
print("뺄셈      :", b - a)
print("스칼라곱  :", 2 * a)
print("원소별 곱 :", a * b)
print("내적      :", np.dot(a, b))
print("a의 크기  :", np.linalg.norm(a))


# %% [Block 3] 7단계 🔵🚀 코사인 유사도 — 길이는 잊고 방향만 본다
import numpy as np

def cosine_similarity(u, v):
    """두 벡터가 얼마나 같은 방향을 보는지 -1~1로 반환합니다."""
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

docs = {
    "기사A (축구)": np.array([5, 4, 0, 0]),
    "기사B (축구)": np.array([10, 8, 1, 0]),
    "기사C (요리)": np.array([0, 1, 6, 5]),
}
names = list(docs)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        u, v = docs[names[i]], docs[names[j]]
        print(f"{names[i]} vs {names[j]} | 내적 {np.dot(u, v):3d} | 코사인 {cosine_similarity(u, v):.4f}")


# %% [Block 4] 8단계 🔵🚀 ML에서 벡터가 쓰이는 곳 — 결국 전부입니다
import numpy as np

emb = {
    "왕":   np.array([0.9, 0.8, 0.1]),
    "남자": np.array([0.1, 0.9, 0.2]),
    "여자": np.array([0.1, 0.1, 0.9]),
    "여왕": np.array([0.9, 0.2, 0.8]),
    "사과": np.array([0.0, 0.1, 0.1]),
}
result = emb["왕"] - emb["남자"] + emb["여자"]
print("왕 - 남자 + 여자 =", np.round(result, 2))

for word, vec in emb.items():
    sim = np.dot(result, vec) / (np.linalg.norm(result) * np.linalg.norm(vec))
    print(f"  {word} 와의 코사인 유사도: {sim:.4f}")
