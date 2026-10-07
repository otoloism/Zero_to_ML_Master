# -*- coding: utf-8 -*-
"""
Zero to 머신러닝 딥러닝 Master — Medium 시리즈 #14
내적·외적·유사도 행렬 — 추천 시스템과 어텐션 QKᵀ의 수학

원문(책): 01-05 「행렬 내적·외적·유사도 행렬」 https://wikidocs.net/439750
Medium : TODO (게시 후 추가)
책     : https://wikidocs.net/book/21464
홈페이지: https://mldict.net
블로그 : https://wikidocs.net/blog/@mldict/

실행: pip install -r requirements.txt && python 14_dot_cross_outer_similarity.py
※ 아래 코드는 책 본문의 코드 블록을 순서대로 옮긴 것입니다(내용 변경 없음).
"""

#======================================================================
# [1] 1단계 — 내적의 두 얼굴: 계산식 vs 각도식
#     성분 곱의 합과 |a||b|cos θ 가 같은 값인지 확인합니다.
#======================================================================
print("\n" + "=" * 60)
print("[1] 1단계 — 내적의 두 얼굴: 계산식 vs 각도식")
print("=" * 60)
import numpy as np

a = np.array([3, 4])
b = np.array([4, 3])

dot = np.dot(a, b)                      # 성분끼리 곱해서 더하기
cos = dot / (np.linalg.norm(a) * np.linalg.norm(b))
angle = np.degrees(np.arccos(cos))

print("내적 a·b      :", dot)
print("|a|, |b|      :", np.linalg.norm(a), np.linalg.norm(b))
print("cos(theta)    :", round(float(cos), 4))
print("사잇각(도)     :", round(float(angle), 2))
print("|a||b|cos(θ)  :", round(float(np.linalg.norm(a) * np.linalg.norm(b) * cos), 4), "← 내적과 같다")


#======================================================================
# [2] 2단계 — 크기의 함정: 내적 vs 코사인
#     점수만 큰 영화 C 때문에 내적이 틀린 결론을 낼 때 코사인이 바로잡는 것을 봅니다.
#======================================================================
print("\n" + "=" * 60)
print("[2] 2단계 — 크기의 함정: 내적 vs 코사인")
print("=" * 60)
import numpy as np

영화A = np.array([5, 1])      # 액션 5, 로맨스 1
영화B = np.array([4, 2])      # 액션 4, 로맨스 2  ← A와 비슷한 취향
영화C = np.array([0, 50])     # 액션 0, 로맨스 50 ← 완전히 다른데 점수만 큼

def 코사인(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

print("내적   A·B =", np.dot(영화A, 영화B), " | A·C =", np.dot(영화A, 영화C))
print("→ 내적만 보면 C가 더 비슷해 보인다:", np.dot(영화A, 영화C) > np.dot(영화A, 영화B))
print()
print("코사인 A,B =", round(float(코사인(영화A, 영화B)), 4),
      "| A,C =", round(float(코사인(영화A, 영화C)), 4))
print("→ 코사인으로 보면 B가 훨씬 비슷하다:", 코사인(영화A, 영화B) > 코사인(영화A, 영화C))


#======================================================================
# [3] 3단계 — 벡터 외적(cross): 방향이 나온다
#     a × b 가 두 벡터에 수직이고, 크기가 평행사변형 넓이이며, 순서를 바꾸면 방향이 반대인지 확인합니다.
#======================================================================
print("\n" + "=" * 60)
print("[3] 3단계 — 벡터 외적(cross): 방향이 나온다")
print("=" * 60)
import numpy as np

a = np.array([2, 3, 4])
b = np.array([5, 6, 7])

c = np.cross(a, b)            # 벡터 외적 — 결과도 벡터(3차원)
print("a × b =", c, " shape:", c.shape)
print("a와 수직인가? a·(a×b) =", np.dot(a, c))
print("b와 수직인가? b·(a×b) =", np.dot(b, c))
print("평행사변형 넓이 |a×b| =", round(float(np.linalg.norm(c)), 4))
print()
x, y = np.array([1, 0, 0]), np.array([0, 1, 0])
print("x축 × y축 =", np.cross(x, y), "← z축이 나온다")
print("y축 × x축 =", np.cross(y, x), "← 순서를 바꾸면 방향이 반대")


#======================================================================
# [4] 4단계 — 외적(outer): 결과는 행렬
#     np.outer 의 shape 과 랭크 1 성질, 내적과의 차이를 확인합니다.
#======================================================================
print("\n" + "=" * 60)
print("[4] 4단계 — 외적(outer): 결과는 행렬")
print("=" * 60)
import numpy as np

a = np.array([1, 2, 3])       # 길이 3
b = np.array([4, 5])          # 길이 2

outer = np.outer(a, b)        # 외적(outer) — 결과는 행렬!
print("a ⊗ b =\n", outer)
print("shape:", outer.shape, "→ (3, 2) 행렬")
print("랭크 :", np.linalg.matrix_rank(outer), "← 항상 1 (모든 행이 b의 배수)")
print()
print("내적 a·a =", np.dot(a, a), "→ 스칼라 (숫자 하나)")
print("외적 shape =", np.outer(a, a).shape, "→ 행렬")


#======================================================================
# [5] 5단계 — 유사도 행렬 XXᵀ
#     모든 쌍의 내적을 한 번에 계산하고 대각선·대칭성을 확인합니다.
#======================================================================
print("\n" + "=" * 60)
print("[5] 5단계 — 유사도 행렬 XXᵀ")
print("=" * 60)
import numpy as np

X = np.array([[5, 1],     # 영화A (액션, 로맨스)
              [4, 2],     # 영화B
              [0, 5]])    # 영화C

S = X @ X.T               # 모든 쌍의 내적을 한 번에!
print("유사도 행렬 S =\n", S)
print("shape:", X.shape, "@", X.T.shape, "→", S.shape)
print()
print("S[0,1] =", S[0, 1], "= A·B =", np.dot(X[0], X[1]))
print("대각선 S[0,0] =", S[0, 0], "= |A|² =", np.dot(X[0], X[0]))
print("대칭인가? S == S.T :", np.array_equal(S, S.T))


#======================================================================
# [6] 6단계 — 코사인 유사도 행렬로 추천
#     행을 정규화한 뒤 X̂X̂ᵀ 로 가장 비슷한 영화를 찾습니다.
#======================================================================
print("\n" + "=" * 60)
print("[6] 6단계 — 코사인 유사도 행렬로 추천")
print("=" * 60)
import numpy as np

X = np.array([[5, 1], [4, 2], [0, 5]])
이름 = ["영화A", "영화B", "영화C"]

크기 = np.linalg.norm(X, axis=1, keepdims=True)   # 각 행의 길이
X_정규화 = X / 크기                                # 길이를 1로 맞춤
C = X_정규화 @ X_정규화.T                          # 코사인 유사도 행렬

print("각 행의 크기:\n", 크기.round(4))
print("\n코사인 유사도 행렬:\n", C.round(3))

기준 = 0
후보 = [(C[기준, j], 이름[j]) for j in range(3) if j != 기준]
print(f"\n{이름[기준]}와 가장 비슷한 영화 →", max(후보)[1], f"(유사도 {max(후보)[0]:.3f})")


#======================================================================
# [7] 7단계 — 어텐션: softmax(QKᵀ/√d)V
#     세 단어 임베딩으로 셀프 어텐션 점수·가중치·출력을 계산합니다.
#======================================================================
print("\n" + "=" * 60)
print("[7] 7단계 — 어텐션: softmax(QKᵀ/√d)V")
print("=" * 60)
import numpy as np

단어 = ["고양이", "야옹", "자동차"]
X = np.array([[3.0, 0.0],      # 고양이 (동물성 3, 기계성 0)
              [2.5, 0.5],      # 야옹
              [0.0, 3.0]])     # 자동차

Q = K = V = X                  # 셀프 어텐션: 셋 다 같은 입력에서 출발
d_k = X.shape[1]

scores = Q @ K.T / np.sqrt(d_k)                    # ① 유사도 행렬 + 스케일
exp = np.exp(scores - scores.max(axis=1, keepdims=True))
weights = exp / exp.sum(axis=1, keepdims=True)     # ② 행마다 합이 1이 되게
output = weights @ V                               # ③ 가중평균으로 섞기

print("① 점수 QKᵀ/√d :\n", scores.round(3))
print("\n② 어텐션 가중치 (행 합 = 1):\n", weights.round(3))
print("   행 합 확인:", weights.sum(axis=1).round(6))
for i, w in enumerate(weights):
    w_others = w.copy()
    w_others[i] = -1                               # 자기 자신은 제외
    top = int(np.argmax(w_others))
    print(f"   '{단어[i]}' → 가장 주목한 다른 단어: '{단어[top]}' ({w[top]:.3f})")
print("\n③ 최종 출력:\n", output.round(3))
