# -*- coding: utf-8 -*-
"""
Zero to 머신러닝 딥러닝 Master — Medium 시리즈 #12
벡터 완전기초 — 화살표로 이해하는 노름·내적·코사인 유사도

원문(책): 01-03 「벡터 — 방향과 크기를 가진 화살표」 https://wikidocs.net/439747
Medium : TODO (게시 후 추가)
책     : https://wikidocs.net/book/21464
홈페이지: https://mldict.net
블로그 : https://wikidocs.net/blog/@mldict/

실행: pip install -r requirements.txt && python 12_vector_basics.py
※ 아래 코드는 책 본문의 코드 블록을 순서대로 옮긴 것입니다(내용 변경 없음).
"""

#======================================================================
# [1] 1단계 — "북쪽으로 3km" vs "그냥 3km": 스칼라와 벡터
#     스칼라와 벡터의 타입·차원·shape 과 벡터의 크기를 비교합니다.
#======================================================================
print("\n" + "=" * 60)
print("[1] 1단계 — '북쪽으로 3km' vs '그냥 3km': 스칼라와 벡터")
print("=" * 60)
import numpy as np

distance = 3                        # 스칼라: 크기만
move = np.array([0, 3])             # 벡터: 크기 + 방향

print("스칼라:", distance, "| 타입:", type(distance).__name__, "| shape: 없음")
print("벡터  :", move, "| 차원:", move.ndim, "| shape:", move.shape)
print("벡터의 크기:", np.linalg.norm(move))


#======================================================================
# [2] 6단계 — 벡터 연산과 내적
#     덧셈·뺄셈·스칼라곱·원소별 곱·내적·노름을 한 번에 확인합니다.
#======================================================================
print("\n" + "=" * 60)
print("[2] 6단계 — 벡터 연산과 내적")
print("=" * 60)
import numpy as np

a = np.array([1, 2])          # 벡터 a
b = np.array([3, 4])          # 벡터 b

print("덧셈      :", a + b)
print("뺄셈      :", b - a)
print("스칼라곱  :", 2 * a)
print("원소별 곱 :", a * b)
print("내적      :", np.dot(a, b))
print("a의 크기  :", np.linalg.norm(a))


#======================================================================
# [3] 7단계 — 코사인 유사도: 길이는 잊고 방향만
#     기사 단어 빈도 벡터로 내적과 코사인 유사도를 비교합니다.
#======================================================================
print("\n" + "=" * 60)
print("[3] 7단계 — 코사인 유사도: 길이는 잊고 방향만")
print("=" * 60)
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


#======================================================================
# [4] 8단계 — ML에서 벡터: 단어 임베딩 연산
#     왕 − 남자 + 여자 결과와 각 단어의 코사인 유사도를 계산합니다.
#======================================================================
print("\n" + "=" * 60)
print("[4] 8단계 — ML에서 벡터: 단어 임베딩 연산")
print("=" * 60)
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
