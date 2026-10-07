# -*- coding: utf-8 -*-
"""
02-03 🔁 3부 — 흐름 제어 조건과 반복

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 02권 🐍 헬로 파이썬 — 파이썬의 탄생부터 NumPy · Pandas  · Plotly까지/02-03 🔁 3부 — 흐름 제어  조건과 반복.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 02-03-K 조건문 — 갈림길에서 길 고르기
def grade_calc(score, attendance):
    """점수와 출석률로 등급을 매깁니다.

    비유:
    학사 규정집처럼 위에서부터 조건을 하나씩 대조해 내려갑니다."""

    # ① 범위 체크 (or 로 하나라도 벗어나면)
    if score < 0 or score > 100 or attendance < 0 or attendance > 100:
        return "입력 값이 유효 범위를 벗어났습니다."

    # ② 과락 먼저 판정 (중첩 조건)
    if score < 40:
        return "결과: 불합격 (과락)"

    # ③ 다중 조건으로 등급 산정 (위에서부터 순서대로!)
    if score >= 90 and attendance >= 80:
        grade = "A"
    elif score >= 80 and attendance >= 75:
        grade = "B"
    elif score >= 70 and attendance >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"

    # ④ 추가 태그 (in 과 not 활용)
    tag = ""
    if grade in ("A", "B") and attendance >= 90:
        tag = " (우수)"
    elif grade == "D" and not (attendance >= 70):
        tag = " (출석 경고)"

    return f"점수: {score}, 출석률: {attendance}% / 최종 등급: {grade}{tag}"

for s, att in [(95, 92), (85, 70), (65, 60), (30, 95)]:
    print(grade_calc(s, att))


# %% [Block 2] 02-03-L 반복문과 컴프리헨션 — 묶어쓰기와 풀어쓰기
guest = 1
while guest < 4:                      # guest가 4 미만인 동안 반복
    print("손님이 {} 명 입니다.".format(guest))
    guest = guest + 1                 # ⚠️ 이 줄이 없으면 무한 루프!
    if guest == 4:
        print("손님이 꽉 찼습니다.")

# 1부터 100까지의 합
num, hap = 1, 0
while num <= 100:
    hap += num                          # hap = hap + num 의 줄임
    num += 1
print("1~100 합:", hap)


# %% [Block 3] 02-03-L 반복문과 컴프리헨션 — 묶어쓰기와 풀어쓰기 — range(시작, 끝, 간격) : 끝은 포함 안 함!
# ── range(시작, 끝, 간격) : 끝은 포함 안 함! ──
for i in range(1, 6):
    print(i, end=' ')          # end=' ' → 줄바꿈 대신 공백
print()

# ── enumerate() : 번호와 값을 함께 꺼내기 ──
langs = ["한국어", "English", "日本語"]
for i, lang in enumerate(langs, start=1):   # start=1 → 1번부터
    print(f"{i}. {lang}")

# ── zip() : 두 목록을 짝지어 꺼내기 ──
days = ["Mon", "Tue", "Wed"]
food = ["Pasta", "Steak", "Cheese", "Toast"]
print(list(zip(days, food)))   # 짧은 쪽(3개)에 맞춰짐!

# ── 중첩 반복문: 구구단 ──
for i in range(2, 4):
    for j in range(1, 4):
        print("{} X {} = {}".format(i, j, i*j), end='  ')
    print()


# %% [Block 4] 02-03-L 반복문과 컴프리헨션 — 묶어쓰기와 풀어쓰기 — continue : 짝수는 건너뛰고 홀수만 더하기
# continue : 짝수는 건너뛰고 홀수만 더하기
hap = 0
for i in range(1, 100):
    if i % 2 == 0:
        continue          # 짝수면 아래를 건너뛰고 다음 i로
    hap += i
print("1~99 홀수의 합:", hap)

# break : 조건을 만나면 즉시 탈출
for i in range(1, 100):
    if i * i > 50:
        print(f"제곱이 50을 넘는 첫 수: {i}")
        break             # 찾았으니 더 볼 필요 없음


# %% [Block 5] 02-03-L 반복문과 컴프리헨션 — 묶어쓰기와 풀어쓰기 — 풀어쓰기
# ── 풀어쓰기 ──
result = []
for num in range(1, 6):
    result.append(num + 5)
print("풀어쓰기:", result)

# ── 묶어쓰기 (똑같은 일을 한 줄로!) ──
result2 = [num + 5 for num in range(1, 6)]
print("묶어쓰기:", result2)

# ── 조건 붙이기: 짝수만 골라 3배 ──
evens = [num * 3 for num in range(1, 11) if num % 2 == 0]
print("짝수×3  :", evens)

# ── 딕셔너리 컴프리헨션도 됩니다 ──
squares = {n: n**2 for n in range(1, 5)}
print("제곱 사전:", squares)
