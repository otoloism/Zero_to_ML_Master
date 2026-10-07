# -*- coding: utf-8 -*-
"""
02-02 📦 2부 — 데이터를 담는 그릇 자료형

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 02권 🐍 헬로 파이썬 — 파이썬의 탄생부터 NumPy · Pandas  · Plotly까지/02-02 📦 2부 — 데이터를 담는 그릇  자료형.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 02-02-D 변수 = 상자가 아니라 "이름표" — 실험 1: 숫자(임뮤터블)
# ── 실험 1: 숫자(임뮤터블) ──
x = 10
y = x               # 같은 10에 이름표 하나 더
y = y + 1           # 11이라는 '새 물건'을 만들고 y를 그쪽으로 옮김
print("x, y:", x, y)  # x는 그대로 10

# ── 실험 2: 리스트(뮤터블) ──
l1 = [1, 2, 3]
l2 = l1             # 같은 리스트에 이름표 하나 더
l2.append(4)       # 물건 '내용'을 직접 수정!
print("l1, l2:", l1, l2)
print("같은 물건인가?", l1 is l2)   # is = 같은 객체인지 확인

# ── 실험 3: 진짜 복사하기 ──
l3 = l1.copy()      # 새 물건을 만들어 내용만 베낌
l3.append(5)
print("l1, l3:", l1, l3)
print("같은 물건인가?", l1 is l3)


# %% [Block 2] 02-02-E 기본 자료형과 연산자 — ① 정수(int) — 소수점 없는 숫자
# ① 정수(int) — 소수점 없는 숫자
age = 25
print(type(age))            # type() = 자료형을 알려주는 함수

# ② 실수(float) — 신경망 가중치는 거의 전부 float!
weight = 0.5
print(type(weight))

# ③ 문자열(str) — 따옴표로 감싼 글자
name = "딥러닝"
print(name + " 시작!")      # + 는 문자열을 이어 붙인다

# ④ 리스트(list) — 대괄호 [], 순서대로 담기
scores = [90, 85, 92, 78]
print(scores[0], len(scores))   # 인덱스는 0부터! len=개수

# ⑤ 딕셔너리(dict) — 중괄호 {}, 이름표로 꺼내기
student = {"이름": "김철수", "나이": 15, "점수": 92}
print(student["이름"])


# %% [Block 3] 02-02-E 기본 자료형과 연산자
print(7 + 3, 7 - 3, 7 * 3, 7 / 3)   # 사칙연산 (/ 는 항상 실수!)
print(7 // 3)     # // 몫  (버림 나눗셈)
print(7 % 3)      # %  나머지 (짝/홀 판정에 필수)
print(7 ** 3)     # ** 거듭제곱 (7의 3제곱)
print(round(7/3, 4))   # round(값, 자릿수) 반올림


# %% [Block 4] 02-02-G 문자열 완전 정복 — 인덱싱 · 슬라이싱 · 포맷 · 메서드
s = "Hello Python"

# ── 인덱싱 & 슬라이싱 ──
print(s[0], s[-1])       # 첫 글자, 마지막 글자
print(s[0:5])            # 0~4번 ("5번 직전까지")
print(s[6:])              # 6번부터 끝까지
print(s[::-1])            # 거꾸로 뒤집기 (유명한 트릭!)

# ── 자주 쓰는 메서드 ──
print(s.upper(), s.lower())       # 대문자/소문자로
print(s.find("Python"))            # 위치 찾기 (없으면 -1)
print(s.replace("Hello", "Hi"))    # 바꾸기
print(s.split(" "))               # 쪼개서 리스트로 (토큰화의 기본!)
print("aaabbc".count("a"), len(s)) # 개수 세기, 전체 길이

# ── 문자열 포맷 3가지 방식 ──
print("{0:*^20}".format("안녕"))  # * 로 채워 가운데 정렬(20칸)
print(f"{3.14159:.2f}")              # f-string: 소수점 2자리 (권장!)


# %% [Block 5] 02-02-H 리스트와 튜플 — 메서드 · 패킹 · 2차원 리스트
lst = [3, 1, 2]

lst.append(4);        print(lst)   # 뒤에 하나 추가
lst.extend([5, 6]);  print(lst)   # 풀어서 추가
lst.insert(1, 99);   print(lst)   # 1번 자리에 끼우기
lst.remove(99);       print(lst)   # 값 99를 찾아 삭제
print(lst.pop(0), lst)          # 0번을 꺼내고 값도 반환
lst.sort();             print(lst)   # 오름차순 정렬
lst.reverse();          print(lst)   # 뒤집기
print(4 in lst)                    # 4가 있는지 확인


# %% [Block 6] 02-02-H 리스트와 튜플 — 메서드 · 패킹 · 2차원 리스트
t = (1, 2, 3)          # 패킹: 셋을 하나로 묶음
a, b, c = t                # 언패킹: 하나를 셋으로 품
print(a, b, c)

def min_max(numbers):
    """최솟값과 최댓값을 함께 돌려줍니다."""
    return min(numbers), max(numbers)   # 사실은 튜플 반환!

low, high = min_max([5, 2, 9])   # 언패킹으로 받기
print(low, high)

# 튜플은 수정 불가! (아래를 실행하면 TypeError)
# t[0] = 99  →  TypeError: 'tuple' object does not support item assignment


# %% [Block 7] 02-02-H 리스트와 튜플 — 메서드 · 패킹 · 2차원 리스트 — 2차원 리스트 = 리스트 안에 리스트 (표 모양!)
# 2차원 리스트 = 리스트 안에 리스트 (표 모양!)
a = [[10, 20], [30, 40], [50, 60]]

for i in range(len(a)):            # 세로 크기(행)
    for j in range(len(a[i])):     # 가로 크기(열)
        print(a[i][j], end=' ')
    print()

# ── 행렬 덧셈: 세 가지 스타일 ──
arr1 = [[1, 2], [2, 3]]
arr2 = [[3, 4], [5, 6]]

# ① 풀어쓰기 (초보자에게 가장 명확)
answer = []
for i in range(len(arr1)):
    row = []
    for j in range(len(arr1[i])):
        row.append(arr1[i][j] + arr2[i][j])
    answer.append(row)
print("① 풀어쓰기 :", answer)

# ② zip 사용 (짝지어 꺼내기)
answer2 = [[x + y for x, y in zip(r1, r2)]
           for r1, r2 in zip(arr1, arr2)]
print("② zip      :", answer2)

# ③ NumPy (7부에서 배울 방법 — 이게 진짜입니다!)
import numpy as np
print("③ NumPy    :", (np.array(arr1) + np.array(arr2)).tolist())


# %% [Block 8] 02-02-I 딕셔너리와 집합
config = {"lr": 0.01, "epochs": 300}

# ── 꺼내는 함수들 ──
print(list(config.keys()))     # 모든 키
print(list(config.values()))   # 모든 값
print(list(config.items()))    # (키, 값) 쌍 — 반복문에서 애용!

# ── get() : 없어도 오류가 안 난다 (중요!) ──
print(config.get("batch_size", 32))  # 없으면 기본값 32
# config["batch_size"]  →  KeyError 발생!

# ── 추가 · 수정 · 삭제 ──
config.update({"batch_size": 64})   # 추가(있으면 수정)
print(config)
print(config.pop("lr"), config)      # 꺼내면서 삭제

# ── 반복문에서 items() 활용 ──
for key, value in config.items():
    print(f"  {key} = {value}")


# %% [Block 9] 02-02-I 딕셔너리와 집합
s1 = set([1, 2, 2, 3])   # 중복 2가 자동으로 사라짐!
s2 = {2, 3, 4}

print(s1)          # 중복 제거 결과
print(s1 & s2)     # & 교집합 (둘 다 있는 것)
print(s1 | s2)     # | 합집합 (전부 모으기)
print(s1 - s2)     # - 차집합 (s1에만 있는 것)

s1.add(99)                 # 추가
s1.discard(99)             # 삭제(없어도 오류 안 남)
print(s1)


# %% [Block 10] 02-02-J 형변환과 초보자가 만나는 3대 에러
print(int("42"))        # 문자 "42" → 숫자 42
print(float("3.14"))    # 문자 → 실수
print(str(123))         # 숫자 → 문자
print(bool(0))          # 0은 False, 나머지 숫자는 True
print(list("abc"))      # 문자열 → 글자 리스트
print(tuple([1, 2]))    # 리스트 → 튜플(봉인)
print(set([1,1,2]))     # 리스트 → 집합(중복 제거)
