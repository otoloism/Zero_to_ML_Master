# -*- coding: utf-8 -*-
"""
04-04-02 자연어와 단어의 분산 표현 — 의미를벡터로

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 04권  신경망과 딥러닝 이론/04-04장 순환 신경망과 자연어 처리/04-04-02 자연어와 단어의 분산 표현 — 의미를벡터로.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 04-04-02-A 원-핫 벡터의 결정적 문제 — "모든 단어가 똑같이 멀다"
import numpy as np

# 사전: [강아지, 고양이, 사과, 자동차]  (총 4개 단어)
cat = np.array([0, 1, 0, 0])  # "고양이" = 2번째 자리만 1
dog = np.array([1, 0, 0, 0])  # "강아지" = 1번째 자리만 1
car = np.array([0, 0, 0, 1])  # "자동차" = 4번째 자리만 1

# 내적(dot product)으로 유사도를 측정해 봅시다
print("고양이·강아지 내적:", np.dot(cat, dog))   # 0 — 전혀 비슷하지 않다?!
print("고양이·자동차 내적:", np.dot(cat, car))   # 0 — 이것도 0?!
print("→ 문제: 모든 단어 쌍의 유사도가 0으로 동일!")


# %% [Block 2] 04-04-02-B 분포 가설 — "친구를 보면 그 사람을 알 수 있다"
import numpy as np

# 작은 말뭉치(corpus): 단어 8개짜리 문장
corpus_text = "you say goodbye and i say hello ."
words = corpus_text.split()
print("말뭉치:", words)

# 단어 → 번호(ID) 대응표 만들기
word_to_id = {}                        # 빈 딕셔너리(사전형)
id_to_word = {}
for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)      # 새 단어마다 순서대로 번호 부여
        word_to_id[word] = new_id
        id_to_word[new_id] = word
print("단어→ID:", word_to_id)

# 말뭉치를 ID 리스트로 변환
corpus_ids = [word_to_id[w] for w in words]
print("말뭉치 ID:", corpus_ids)

# 동시발생 행렬 만들기 (윈도우=1: 바로 양옆만)
vocab_size = len(word_to_id)
co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

for idx, word_id in enumerate(corpus_ids):
    if idx - 1 >= 0:                   # 왼쪽 이웃이 있으면
        co_matrix[word_id, corpus_ids[idx-1]] += 1
    if idx + 1 < len(corpus_ids):    # 오른쪽 이웃이 있으면
        co_matrix[word_id, corpus_ids[idx+1]] += 1

print("\n동시발생 행렬:")
print(co_matrix)


# %% [Block 3] 04-04-02-C 코사인 유사도 — 나침반 방향이 얼마나 같은가?
def cos_similarity(x, y, eps=1e-8):
    """코사인 유사도: 두 벡터의 방향이 얼마나 비슷한가?
    비유: '나침반 방향 비교기' — 같은 방향이면 1, 직각이면 0"""
    nx = x / (np.sqrt(np.sum(x**2)) + eps)   # 단위벡터로 정규화
    ny = y / (np.sqrt(np.sum(y**2)) + eps)
    return np.dot(nx, ny)                    # 정규화된 벡터의 내적

# 동시발생 행렬에서 단어 벡터 꺼내기
you_vec     = co_matrix[word_to_id['you']]
i_vec       = co_matrix[word_to_id['i']]
goodbye_vec = co_matrix[word_to_id['goodbye']]

print("you vs i:",       round(cos_similarity(you_vec, i_vec), 3))
print("you vs goodbye:", round(cos_similarity(you_vec, goodbye_vec), 3))


# %% [Block 4] 04-04-02-D PPMI — "단순히 자주 나온다고 중요한 건 아니다"
def ppmi(C, verbose=False, eps=1e-8):
    """PPMI 행렬 계산
    비유: '진짜 친구 감별기' — 우연히 만난 것 vs 의미 있는 만남 구별"""
    M = np.zeros_like(C, dtype=np.float64)
    N = np.sum(C)                          # 전체 동시발생 횟수 합
    S = np.sum(C, axis=1)                   # 각 단어의 등장 횟수

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i,j] * N / (S[i]*S[j] + eps) + eps)
            M[i,j] = max(0, pmi)           # 음수는 0으로 (PPMI)
    return M

W = ppmi(co_matrix)
np.set_printoptions(precision=2)
print("PPMI 행렬:")
print(W)


# %% [Block 5] 04-04-02-E SVD로 차원 축소 — "300페이지를 3페이지로 요약" — SVD 실행: W = U × S × V^T
# SVD 실행: W = U × S × V^T
U, S, V = np.linalg.svd(W)

# U의 각 행이 각 단어의 "압축된 의미 벡터"
# 앞 2차원만 사용하면 2D 좌표로 시각화 가능!
print("'you'의 원래 벡터 (7차원):", W[word_to_id['you']].round(2))
print("'you'의 압축 벡터 (2차원):", U[word_to_id['you']][::2].round(2))
