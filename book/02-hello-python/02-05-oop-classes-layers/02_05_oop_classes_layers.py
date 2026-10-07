# -*- coding: utf-8 -*-
"""
02-05 🏗️ 5부 — 객체지향 클래스에서 Layer까지

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 02권 🐍 헬로 파이썬 — 파이썬의 탄생부터 NumPy · Pandas  · Plotly까지/02-05 🏗️ 5부 — 객체지향  클래스에서 Layer까지.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 02-05-R 시퀀스 · 이터레이터 · 제너레이터
import sys

# ── 이터레이터: iter() 로 만들고 next() 로 꺼낸다 ──
it = iter([1, 2, 3])
print(next(it), next(it), next(it))

# ── 제너레이터: yield 로 '하나씩 내주는' 함수 ──
def count_up(n):
    """1부터 n까지 10배 해서 하나씩 내줍니다.

    비유:
    return은 '요리를 완성해 한 번에 내놓기',
    yield는 '뷔페처럼 한 접시씩 계속 내주기'입니다."""
    for i in range(1, n + 1):
        yield i * 10          # return이 아니라 yield!

g = count_up(3)
print(next(g), next(g), next(g))

# ── 메모리 비교: 100만 개를 다룰 때 ──
lst = [i*i for i in range(1000000)]   # 리스트: 전부 만들어 저장
gen = (i*i for i in range(1000000))   # 제너레이터: 소괄호!
print(f"리스트   : {sys.getsizeof(lst):,} bytes")
print(f"제너레이터: {sys.getsizeof(gen):,} bytes")


# %% [Block 2] 02-05-S 클래스 — "설계도"의 모든 것
class Dog:
    """강아지를 표현하는 클래스.

    비유:
    '강아지 설계도'입니다. 이름(데이터)과 짖는 기능(메서드)을
    하나로 묶어 두었습니다."""

    def __init__(self, name, age):
        # 객체가 만들어질 때 자동 실행 (초기화)
        self.name = name          # self = "이 강아지의"
        self.age = age

    def bark(self):
        """이 강아지가 짖습니다."""
        return f"{self.name}: 멍멍!"

    def birthday(self):
        """나이를 한 살 올립니다."""
        self.age += 1
        return f"{self.name}는 이제 {self.age}살!"

# 설계도로 실제 강아지(인스턴스) 만들기
my_dog = Dog("초코", 3)
your_dog = Dog("보리", 5)      # 같은 틀, 다른 속재료

print(my_dog.bark())
print(your_dog.bark())
print(my_dog.birthday())
print(f"보리 나이는 그대로: {your_dog.age}")


# %% [Block 3] 02-05-T 매직 메서드 — 파이썬 문법과 내 클래스를 연결하기
class Point:
    """좌표를 관리하는 클래스.

    비유:
    내부 데이터를 튜플(봉인된 선반)로 보관해 실수로 바뀌는 것을 막고,
    매직 메서드로 '파이썬 표준 플러그'를 달아둡니다."""

    def __init__(self, *args):
        self._coords = tuple(args)     # 튜플로 저장 → 불변성 확보

    def __repr__(self):
        return f"Point{self._coords}"     # 출력 모양 지정

    def __len__(self):
        return len(self._coords)         # len(p) 가능해짐

    def __getitem__(self, index):
        return self._coords[index]      # p[0] 가능해짐

    def __add__(self, other):
        return Point(*(a + b for a, b in zip(self._coords, other._coords)))

    def __sub__(self, other):
        return Point(*(a - b for a, b in zip(self._coords, other._coords)))

    def __eq__(self, other):
        return self._coords == other._coords

    def __ge__(self, other):
        return sum(self._coords) >= sum(other._coords)

p1 = Point(10, 20)
p2 = Point(5, 5)

print(p1)                # __repr__ 작동
print(p1[0])             # __getitem__ 작동
print(len(p1))           # __len__ 작동
print(p1 + p2)           # __add__ 작동
print(p1 - p2)           # __sub__ 작동
print(p1 >= p2)          # __ge__ 작동
print(p1 == Point(10, 20))  # __eq__ 작동


# %% [Block 4] 02-05-U 메서드 3종과 상속 — 인스턴스 · 클래스 · 스태틱
class Counter:
    """만들어진 개수를 세는 클래스."""

    total = 0                       # 클래스 변수 (모든 인스턴스가 공유)

    def __init__(self, name):
        self.name = name              # 인스턴스 변수 (각자 다름)
        Counter.total += 1            # 가게 전체 카운터 증가

    def hello(self):                  # ① 인스턴스 메서드 (self)
        return f"안녕, {self.name}"

    @classmethod
    def how_many(cls):                # ② 클래스 메서드 (cls)
        return f"총 {cls.total}개"

    @staticmethod
    def add(x, y):                   # ③ 스태틱 메서드 (아무것도 없음)
        return x + y

c1 = Counter("A")
c2 = Counter("B")

print(c1.hello())              # 개별 객체의 데이터
print(Counter.how_many())      # 클래스 전체 데이터
print(Counter.add(3, 4))       # 객체 없이도 호출 가능


# %% [Block 5] 02-05-U 메서드 3종과 상속 — 인스턴스 · 클래스 · 스태틱
class Animal:
    """모든 동물의 기본 설계도 (부모 클래스)"""

    def __init__(self, name):
        self.name = name

    def speak(self):
        return "..."          # 자식이 고쳐 쓸 자리

    def introduce(self):
        return f"저는 {self.name}, {self.speak()}"

class Dog(Animal):          # 괄호 안에 부모 = 상속!
    """Animal을 물려받은 강아지"""

    def speak(self):
        return "멍멍!"        # 부모 메서드를 '덮어쓰기'(오버라이드)

class Cat(Animal):
    """Animal을 물려받은 고양이"""

    def __init__(self, name, indoor=True):
        super().__init__(name)   # 부모의 초기화를 먼저 실행!
        self.indoor = indoor

    def speak(self):
        return "야옹~"

for pet in [Dog("초코"), Cat("나비")]:
    print(pet.introduce())      # introduce는 물려받아 그대로 씀


# %% [Block 6] 02-05-V 🏁 함수와 클래스가 신경망 Layer가 되기까지
import numpy as np

class Layer:
    """모든 층의 '기본 설계도' (부모 클래스).

    비유:
    프랜차이즈 본사의 '표준 매장 규격서'입니다.
    모든 지점은 반드시 forward 라는 이름의 창구를 가져야 합니다."""

    def __init__(self, name):
        self.name = name

    def forward(self, x):
        """자식 클래스가 반드시 구현해야 하는 규격"""
        raise NotImplementedError

    def __call__(self, x):
        """매직 메서드! layer(x) 라고 쓰면 forward가 실행됩니다.

        비유:
        매장 문(__call__)으로 들어가면 자동으로 창구(forward)로 안내됩니다."""
        return self.forward(x)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"

class Affine(Layer):
    """선형 변환 층: y = xW + b  (🔗 01-13 2단계)

    비유:
    '원재료를 가공하는 기계'입니다.
    자기 설정값(W, b)을 스스로 들고 있습니다."""

    def __init__(self, n_in, n_out, seed=0):
        super().__init__("affine")          # 부모 초기화 먼저!
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.5, (n_in, n_out))  # 🧠 가중치를 스스로 보관
        self.b = np.zeros(n_out)

    def forward(self, x):
        self.x = x                            # 🧠 역전파용으로 입력 기억!
        return x @ self.W + self.b

class Relu(Layer):
    """활성화 층: 음수는 0으로 (🔗 01-13 2단계)

    비유:
    '불량품을 걸러내는 검수대'입니다."""

    def __init__(self):
        super().__init__("relu")

    def forward(self, x):
        self.mask = (x <= 0)                  # 🧠 어디를 껐는지 기억(역전파용)
        out = x.copy()                      # ⚠️ copy! 원본 보호 (O단원)
        out[self.mask] = 0
        return out

# ===== 레고처럼 이어 붙이기 =====
x = np.array([1.0, 0.5])

layers = [Affine(2, 3), Relu(), Affine(3, 2)]

print("층 구성:", layers)

out = x
for layer in layers:
    out = layer(out)                       # __call__ 덕분에 이렇게 간결!
    print(f"  {layer.name:8s} 통과 → {np.round(out, 4)}")

print("최종 출력:", np.round(out, 4))
