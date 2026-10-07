# -*- coding: utf-8 -*-
"""
03-08-01 L1 노름과 L2 노름 (L1 Norm & L2 Norm)

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-08장 - 정규화 심화 및 대규모 학습/03-08-01 L1 노름과 L2 노름.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 7단원 — NumPy로 직접 구현하기
import numpy as np

def l1_norm(x):
    """L1 노름: 각 요소의 절댓값을 전부 더한다."""
    x_norm = np.abs(x)        # np.abs : 부호를 떼고 크기만 남긴다 (-3 -> 3)
    x_norm = np.sum(x_norm)   # np.sum : 배열의 모든 값을 더한다 (수식의 시그마 기호)
    return x_norm

def l2_norm(x):
    """L2 노름: 제곱해서 더한 뒤 제곱근을 씌운다."""
    x_norm = x * x            # 원소끼리 곱셈. x**2 와 같다.
                              #   NumPy에서 * 는 행렬곱이 아니라 '자리마다 곱하기'다.
    x_norm = np.sum(x_norm)   # 다 더한다
    x_norm = np.sqrt(x_norm)  # np.sqrt : 제곱근(루트). 제곱의 반대 연산
    return x_norm

x = np.array([1, 2, 3, 4, 5])

print("x        =", x)
print("L1 노름  =", l1_norm(x))
print("L2 노름  =", l2_norm(x))


# %% [Block 2] sklearn·NumPy 내장 함수와 대조 검산 — 직접 만든 함수가 맞는지, NumPy 내장 함수와 비교해 본다.
# 직접 만든 함수가 맞는지, NumPy 내장 함수와 비교해 본다.
#   np.linalg.norm(x, ord=1) : ord가 p에 해당한다
print("내가 만든 L1 :", l1_norm(x))
print("NumPy   L1  :", np.linalg.norm(x, ord=1))
print()
print("내가 만든 L2 :", l2_norm(x))
print("NumPy   L2  :", np.linalg.norm(x, ord=2))
print("NumPy 기본값 :", np.linalg.norm(x))   # ord를 안 쓰면 L2가 기본
print()

# p를 바꿔가며 노름 값이 어떻게 변하는지 확인
for p in [1, 2, 3, 10]:
    print("L%-2d 노름 = %.4f" % (p, np.linalg.norm(x, ord=p)))
print("L-inf 노름 = %.4f  (= 최댓값)" % np.linalg.norm(x, ord=np.inf))


# %% [Block 3] 규제 항으로 쓸 때의 실제 모습 — 실제 규제에서는 '가중치 벡터'의 노름을 손실에 더한다.
# 실제 규제에서는 '가중치 벡터'의 노름을 손실에 더한다.
w = np.array([3.0, -1.5, 0.0, 0.8, -2.2])   # 학습된 가중치라고 가정

lam = 0.1   # 규제 강도 lambda (다음 절의 주인공)

l1_penalty = lam * np.sum(np.abs(w))        # Lasso가 더하는 과태료
l2_penalty = lam * np.sum(w ** 2)           # Ridge가 더하는 과태료
                                            #   ** 는 거듭제곱 연산자 (w**2 = w의 제곱)

print("가중치 w      :", w)
print("L1 페널티     : %.4f" % l1_penalty)
print("L2 페널티     : %.4f" % l2_penalty)
print()
print("→ 큰 가중치(3.0)가 L2에서 얼마나 기여하나:")
print("   L1 기여분 : %.2f (전체의 %.0f%%)" % (abs(w[0]), 100*abs(w[0])/np.sum(np.abs(w))))
print("   L2 기여분 : %.2f (전체의 %.0f%%)" % (w[0]**2, 100*w[0]**2/np.sum(w**2)))


# %% [Block 4] 실습 — 실습 1 정답
# 실습 1 정답
v = np.array([-3, 4])
print("L1 =", l1_norm(v), "  (|-3| + |4| = 3 + 4)")
print("L2 =", l2_norm(v), "  (√(9+16) = √25)")


# %% [Block 5] 실습 — 실습 3 정답 — "몰빵 벡터" vs "고른 벡터"
# 실습 3 정답 — "몰빵 벡터" vs "고른 벡터"
a = np.array([10.0, 0.0, 0.0, 0.0])      # 한 곳에 몰빵
b = np.array([2.5, 2.5, 2.5, 2.5])       # 고르게 분산

print("%-12s %8s %8s" % ("벡터", "L1", "L2"))
print("-" * 30)
print("%-12s %8.2f %8.2f" % ("몰빵 [10,0,0,0]", l1_norm(a), l2_norm(a)))
print("%-12s %8.2f %8.2f" % ("고른 [2.5]*4  ", l1_norm(b), l2_norm(b)))
print()
print("→ L1은 둘 다 10으로 '똑같다'고 본다.")
print("→ L2는 몰빵을 10.00, 고른 쪽을 5.00으로 본다. 몰빵을 2배 더 나쁘게 본다!")
print("→ 그래서 L2 규제는 가중치를 '고르게 펴는' 성질(weight sharing)을 가진다.")
