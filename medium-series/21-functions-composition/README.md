# #21 함수와 합성함수 — 딥러닝은 함수를 수백 겹 쌓은 것

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/21-functions-composition/21_functions_composition.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #21의 실습 코드입니다.
> 원문: 책 **01-08 「함수와 합성함수 — 딥러닝의 구조」**

합성함수 g(f(x))와 f(g(x))가 다르다는 것, 신경망이 층(함수)을 겹겹이 합성한 것이라는 것, 활성화 함수 없이 쌓은 선형 층은 행렬 하나로 붕괴한다는 것을 확인하고, sigmoid·ReLU·tanh, 연쇄법칙(수치미분 검산), 기울기 소실(0.25ⁿ)과 ReLU의 해법까지 코드로 따라갑니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **https://medium.com/p/3fe59908fc90** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [01-08 함수와 합성함수 — 딥러닝의 구조](https://wikidocs.net/439745) |
| ✍️ 블로그 글 | [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/) (이 편의 블로그 글은 아직 없음) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`21_functions_composition.py`](21_functions_composition.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`21_functions_composition.ipynb`](21_functions_composition.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`) |

## 🧭 코드 순서

1. 1단원 — 합성함수와 순서의 중요성
2. 1단원 — 층을 함수로 이어 붙이기
3. 2단원 — 선형 두 층 = 선형 한 층
4. 2단원 — sigmoid, ReLU, tanh
5. 3단원 — 연쇄법칙 dy/dx = dy/du × du/dx
6. 3단원 — 연쇄법칙 결과를 수치미분으로 검산
7. 4단원 — 층이 깊어질수록 기울기가 사라진다
8. 4단원 — ReLU 미분은 곱해도 1

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 21_functions_composition.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
