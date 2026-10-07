# -*- coding: utf-8 -*-
"""
02-04 🧠 4부 — 함수와 메모리 파이썬의 심장

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 02권 🐍 헬로 파이썬 — 파이썬의 탄생부터 NumPy · Pandas  · Plotly까지/02-04 🧠 4부 — 함수와 메모리  파이썬의 심장.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 02-04-M 함수 — "레시피"의 모든 것
def add(a, b):
    """두 숫자를 더합니다.

    비유:
    재료 두 개를 넣으면 합쳐진 결과가 나오는 믹서기입니다.

    Args:
        a (int): 첫 번째 숫자
        b (int): 두 번째 숫자

    Returns:
        int: 두 숫자의 합
    """
    return a + b

def greet(name, greeting="안녕하세요"):    # 기본값 있는 매개변수
    """인사말을 만듭니다. greeting을 생략하면 '안녕하세요'."""
    return f"{greeting}, {name}님!"

print(add(3, 5))
print(greet("철수"))                        # 기본값 사용
print(greet("영희", "반갑습니다"))            # 포지셔널 인자
print(greet(greeting="하이", name="민수"))    # 키워드 인자(순서 무관!)
print(add.__doc__.splitlines()[0])        # docstring 읽기


# %% [Block 2] 02-04-M 함수 — "레시피"의 모든 것
def save_winner(*args, **kwargs):
    """우승자를 기록합니다.

    비유:
    *args는 '순서대로 담는 봉지'(튜플),
    **kwargs는 '이름표 붙여 담는 상자'(딕셔너리)입니다."""
    print("args  (튜플)   :", args)
    print("kwargs(딕셔너리):", kwargs)
    if kwargs.get("name1"):
        print("  1등:", kwargs["name1"])

save_winner("홍길동", "가가멜", name1="철수", age=15)


# %% [Block 3] 02-04-M 함수 — "레시피"의 모든 것
def get_input_user(msg, casting=int):
    """사용자에게 msg를 보여주고, casting 형태로 변환해 돌려줍니다.

    비유:
    은행 창구 직원처럼, 잘못된 서류를 받으면
    "다시 작성해 주세요"라고 돌려보내고 올바를 때만 접수합니다.

    Args:
        msg (str): 입력 시 보여줄 문구
        casting (type): 변환할 자료형 (int, float, str 등)

    Returns:
        casting 타입으로 변환된 사용자 입력값
    """
    while True:
        user_input = input(msg)

        # 이름(str)인데 공백만 넣었다면 다시 받기
        if casting is str and not user_input.strip():
            print("이름이 입력되지 않았습니다.")
            continue

        try:
            return casting(user_input)      # 변환 성공 → 반환
        except ValueError:
            print("잘못된 입력입니다. 다시 입력하세요.")

# 사용 예 (실제 실행 시 키보드 입력을 기다립니다)
# name = get_input_user("이름을 입력하세요> ", str)
# age  = get_input_user("나이를 입력하세요> ")      # casting=int 생략


# %% [Block 4] 02-04-N 스코프와 메모리 스택 — 개인 사물함 vs 공용 사물함
a = 10
b = [1, 2, 3, 4]

def function():
    a = 20                   # ← 함수 스택에 '새로운' a를 만듦!
    b = [5, 6, 7, 8]      # ← 함수 스택에 '새로운' b를 만듦!

function()
print(a)
print(b)


# %% [Block 5] 02-04-N 스코프와 메모리 스택 — 개인 사물함 vs 공용 사물함
a = 10
b = [1, 2, 3, 4]

def function():
    global a                 # "a는 공용 사물함 걸 쓸게요" 선언
    a = 20                   # 이제 전역 a가 진짜로 바뀜
    b.extend([5, 6, 7, 8])  # ⚠️ global 없이도 바뀜! (아래 설명)

function()
print(a)
print(b)


# %% [Block 6] 02-04-N 스코프와 메모리 스택 — 개인 사물함 vs 공용 사물함
a = 10

def function():
    print(a)        # ← 여기서 오류! "a가 아직 없다"
    a = 20          # ← 이 줄 때문에 a가 '지역변수'로 판정됨

function()


# %% [Block 7] 02-04-O 뮤터블 vs 임뮤터블 — 파이썬이 왜 이렇게 설계됐나
a = 10
b = [1, 2, 3, 4]
print("시작       :", a, b)

def function_a(c, d):
    """재할당만 하는 함수 — 바깥에 영향 없음"""
    c = 20                    # 이름표 이동 (함수 안에서만)
    d = [5, 6, 7, 8]       # 이름표 이동 (함수 안에서만)

function_a(a, b)
print("재할당 후  :", a, b)

def function_b(c, d):
    """내용을 수정하는 함수 — 바깥까지 바뀜!"""
    c = 30                    # 숫자는 임뮤터블 → 영향 없음
    d.extend([9, 10])       # 리스트는 뮤터블 → 바깥도 바뀜!

function_b(a, b)
print("내용수정 후:", a, b)


# %% [Block 8] 02-04-O 뮤터블 vs 임뮤터블 — 파이썬이 왜 이렇게 설계됐나
l1 = [1, 2, 3]
l2 = [1, 2, 3]        # 내용은 같지만 '다른 물건'

print("l1 == l2 :", l1 == l2)   # 내용 비교 → True
print("l1 is l2 :", l1 is l2)   # 정체성 비교 → False


# %% [Block 9] 02-04-P 참조 카운트 · 가비지 컬렉터 · GIL
import sys

obj = [1]                                # 물건 생성, 이름표 obj 하나
print("초기      :", sys.getrefcount(obj))

ref = obj                                # 이름표 ref 추가 → +1
print("참조 추가 :", sys.getrefcount(obj))

del ref                                  # 이름표 제거 → -1
print("참조 제거 :", sys.getrefcount(obj))


# %% [Block 10] 02-04-Q 일급 함수 · 클로저 · 람다 — 함수도 물건이다
def hi():
    return "Hello"

hello = hi                  # 괄호 없이! 함수 '자체'에 이름표 하나 더
print(hello(), type(hello))

# 딕셔너리에 함수를 담아 '계산기' 만들기
ops = {
    '+': lambda x, y: x + y,
    '*': lambda x, y: x * y,
}
print(ops['+'](3, 4), ops['*'](3, 4))


# %% [Block 11] 02-04-Q 일급 함수 · 클로저 · 람다 — 함수도 물건이다
def make_adder(x):
    """x를 기억하는 덧셈 로봇을 만들어 돌려줍니다."""
    def adder(y):          # 안쪽 함수가 바깥의 x를 사용
        return x + y
    return adder            # 함수 '자체'를 반환 (괄호 없음!)

add_5 = make_adder(5)      # 5를 기억하는 로봇
add_10 = make_adder(10)    # 10을 기억하는 로봇

print(add_5(3), add_10(3))

# 로봇 안의 '기억 칸'을 직접 들여다보기
print("add_5가 기억한 값:", add_5.__closure__[0].cell_contents)


# %% [Block 12] 02-04-Q 일급 함수 · 클로저 · 람다 — 함수도 물건이다 — 람다 기본
# ── 람다 기본 ──
double = lambda v: v * 2
print(double(5))

# ── map: 모든 원소에 함수 적용 ──
print(list(map(int, ["1", "2", "3"])))          # 문자→숫자 일괄 변환
print(list(map(lambda v: v**2, [1,2,3])))   # 제곱

# ── filter: 조건에 맞는 것만 남기기 ──
print(list(filter(lambda v: v > 1, [1,2,3])))

# ── 데코레이터: 함수를 감싸 기능 추가 (일급함수+클로저의 응용) ──
def logger(func):
    """함수 호출을 기록하는 껍데기를 씌웁니다.

    비유:
    선물 상자에 '포장지'를 한 겹 두르는 것과 같습니다.
    내용물(원래 함수)은 그대로인데 겉모습에 기능이 붙습니다."""
    def wrapper(*args, **kwargs):
        print(f"[호출] {func.__name__}{args}")
        result = func(*args, **kwargs)
        print(f"[반환] {result}")
        return result
    return wrapper

@logger                      # add = logger(add) 와 같은 뜻
def add(a, b):
    return a + b

add(3, 5)
