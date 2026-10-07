# -*- coding: utf-8 -*-
"""
Zero to 머신러닝 딥러닝 Master — Medium 시리즈 #03
벡터란? 장바구니로 이해하는 벡터·내적·코사인 유사도 (NumPy 실습)

원문(책): 00-01 「벡터 — 장바구니 속 숫자의 줄서기」 https://wikidocs.net/439839
Medium : https://medium.com/p/9a7f826c3b2e
책     : https://wikidocs.net/book/21464
홈페이지: https://mldict.net
블로그 : https://wikidocs.net/blog/@mldict/

실행: pip install -r requirements.txt && python 03_vectors.py
※ 아래 코드는 책 본문의 코드 블록을 순서대로 옮긴 것입니다(내용 변경 없음).
"""

#======================================================================
# [1] 1단계 — 벡터란 무엇인가?
#     장바구니 [사과, 바나나, 우유] 를 NumPy 배열(벡터)로 만들고 .shape, 인덱싱, dtype 을 확인합니다.
#======================================================================
print("\n" + "=" * 60)
print("[1] 1단계 — 벡터란 무엇인가?")
print("=" * 60)
import numpy as np   # 수치 연산 라이브러리를 np라는 별칭으로 불러옵니다

# 자리 약속: [0]=사과, [1]=바나나, [2]=우유  ← 반드시 주석으로 남긴다!
# np.array()는 파이썬 리스트를 "벡터 연산이 가능한" 배열로 바꿔줍니다
basket = np.array([3, 2, 1])

print("벡터 내용 :", basket)         # 벡터 전체 출력
print("차원 개수 :", basket.shape)   # (3,) → 숫자가 3개(3차원)
print("첫째 성분 :", basket[0])     # 사과 개수 = 3 (0번부터 셈)
print("자료형    :", basket.dtype)   # int64 (정수형)


#======================================================================
# [2] 2단계 — 벡터의 덧셈과 뺄셈
#     같은 자리끼리 더하고 뺍니다. 계산 전에 assert 로 차원을 확인하는 습관도 함께.
#======================================================================
print("\n" + "=" * 60)
print("[2] 2단계 — 벡터의 덧셈과 뺄셈")
print("=" * 60)
import numpy as np

# 자리 약속: [0]=사과, [1]=바나나
monday  = np.array([3, 2])   # 사과 3, 바나나 2
tuesday = np.array([1, 4])   # 사과 1, 바나나 4

# 계산 전 차원 확인 — 습관화하면 버그의 90%가 사라집니다
assert monday.shape == tuesday.shape, "차원이 다르면 더할 수 없습니다!"

# '+' 는 __add__ 를 호출해 같은 자리끼리 더해줍니다
weekly_total = monday + tuesday
print("이번 주 합계   :", weekly_total)

# 뺄셈: "목표 - 현재" = 앞으로 줄여야 할 양
# 자리 약속: [0]=몸무게(kg), [1]=체지방률(%)
goal    = np.array([65, 15])
current = np.array([72, 20])
gap = goal - current            # __sub__ 호출
print("목표까지 남은 차이:", gap)


#======================================================================
# [3] 3단계 — 스칼라 곱
#     숫자 하나를 모든 성분에 곱합니다. 벡터끼리의 * 는 "성분별 곱"이라는 점에 주의!
#======================================================================
print("\n" + "=" * 60)
print("[3] 3단계 — 스칼라 곱")
print("=" * 60)
import numpy as np

# 자리 약속: [0]=계란(개), [1]=밀가루(g)
recipe = np.array([2, 100])

recipe_x3   = 3 * recipe     # __rmul__ 호출 → 모든 성분에 3을 곱함
recipe_half = 0.5 * recipe   # 절반 레시피 (결과가 실수형으로 바뀜!)
recipe_neg  = -1 * recipe    # 방향 반전 (화살표가 정반대)

print("3배 레시피 :", recipe_x3,   "dtype:", recipe_x3.dtype)
print("절반 레시피:", recipe_half, "dtype:", recipe_half.dtype)
print("반전 레시피:", recipe_neg)

# ⚠️ 주의: 벡터끼리의 '*' 는 스칼라 곱이 아니라 "성분별 곱"입니다
print("성분별 곱  :", np.array([2, 3]) * np.array([10, 100]))


#======================================================================
# [4] 4단계 — 벡터의 크기(Norm)
#     √(성분² 의 합)을 직접 구현한 값과 np.linalg.norm 을 비교하고, 단위벡터로 정규화합니다.
#======================================================================
print("\n" + "=" * 60)
print("[4] 4단계 — 벡터의 크기(Norm)")
print("=" * 60)
import numpy as np

v = np.array([3, 4])

# 방법 1) 공식을 그대로 구현: √(3² + 4²)
manual = np.sqrt(np.sum(v ** 2))

# 방법 2) 전용 함수 (실무 권장)
builtin = np.linalg.norm(v)

print("직접 계산 :", manual)    # 5.0
print("norm 함수 :", builtin)   # 5.0

# 3차원도 규칙은 동일: √(1+4+4) = √9 = 3
print("3차원 크기:", np.linalg.norm(np.array([1, 2, 2])))

# 음수가 있어도 제곱하면 양수 → 크기는 절대 음수가 될 수 없음
print("(-3,4) 크기:", np.linalg.norm(np.array([-3, 4])))

# 정규화: 크기로 나누면 길이가 1인 "방향만 남은" 단위벡터가 됩니다
unit = v / np.linalg.norm(v)
print("단위벡터  :", unit, "| 그 크기:", np.linalg.norm(unit))


#======================================================================
# [5] 5단계 — 내적(Dot Product)과 코사인 유사도
#     영화 취향 벡터로 내적과 코사인 유사도를 계산합니다 — 추천 시스템의 핵심 원리.
#======================================================================
print("\n" + "=" * 60)
print("[5] 5단계 — 내적(Dot Product)과 코사인 유사도")
print("=" * 60)
import numpy as np

# 자리 약속: [0]=액션, [1]=로맨스, [2]=공포
movie_a = np.array([5, 1, 0])   # 액션 영화
movie_b = np.array([4, 2, 1])   # 액션 위주
movie_c = np.array([0, 5, 0])   # 순수 로맨스

print("A·B 내적:", np.dot(movie_a, movie_b))  # 20+2+0 = 22
print("A·C 내적:", np.dot(movie_a, movie_c))  # 0+5+0 = 5

def cosine_similarity(x, y, eps=1e-10):
    """
    두 벡터가 얼마나 비슷한 '방향'을 가리키는지 -1~1 사이 값으로 반환합니다.

    비유:
    두 사람이 걸어가는 방향이 같으면 1,
    직각이면 0, 정반대면 -1.
    (얼마나 '빨리' 걷는지=크기 는 무시하고 방향만 봅니다)
    """
    dot_value  = np.dot(x, y)                              # 분자: 내적
    norm_product = np.linalg.norm(x) * np.linalg.norm(y)   # 분모: 크기의 곱
    if norm_product < eps:      # 영벡터 방어 → 0으로 나누기 방지
        return 0.0
    return dot_value / norm_product

print("A-B 유사도:", round(cosine_similarity(movie_a, movie_b), 3))
print("A-C 유사도:", round(cosine_similarity(movie_a, movie_c), 3))
print("영벡터 방어:", cosine_similarity(movie_a, np.array([0, 0, 0])))


#======================================================================
# [6] 심화 🔴 — 브로드캐스트와 벡터화
#     스칼라가 자동으로 늘어나는 브로드캐스트와, for문 vs 벡터화 속도 비교 (실행 시간은 환경마다 다릅니다).
#======================================================================
print("\n" + "=" * 60)
print("[6] 심화 🔴 — 브로드캐스트와 벡터화")
print("=" * 60)
import numpy as np
import time

scores = np.array([70, 80, 90])
print("전원 10점 가산:", scores + 10)   # 브로드캐스트

# 속도 비교: 100만 개 원소의 합
big = np.arange(1_000_000)

t0 = time.time(); total_loop = 0
for x in big:            # 느린 방식: 파이썬 for문
    total_loop += x
t1 = time.time(); total_vec = big.sum()   # 빠른 방식: 벡터화
t2 = time.time()

print("for문  :", round(t1 - t0, 4), "초")
print("벡터화 :", round(t2 - t1, 4), "초")
