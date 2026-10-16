# #23 · 1부 편미분·그래디언트 쉽게 이해하기 — 안개 산에서 발밑 경사만으로 하산하기 (1부)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/23-1-partial-derivative-gradient/23_1_partial_derivative_gradient.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #23 · 1부의 실습 코드입니다.
> 원문: 책 **01-10 「편미분·그래디언트·경사하강법 (1부)」**

f(x, y) = x² + 2y² 의 손실과 그래디언트 ∇f = (2x, 4y) 를 구현하고, (4, 3)에서 출발하는 경사하강법으로 손실이 34 → 0.02 로 줄어드는 과정을 표로 확인한 뒤, 같은 규칙을 PyTorch 식 옵티마이저 클래스로 감쌉니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [01-10 편미분·그래디언트·경사하강법 (1부)](https://wikidocs.net/439754) |
| ✍️ 블로그 글 | [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/) (이 편의 블로그 글은 아직 없음) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`23_1_partial_derivative_gradient.py`](23_1_partial_derivative_gradient.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`23_1_partial_derivative_gradient.ipynb`](23_1_partial_derivative_gradient.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`) |

## 🧭 코드 순서

1. 01-10-C 3단계 — 손실과 그래디언트를 코드로 옮기기
2. 01-10-E 5단계 — 경사하강법으로 하산하기
3. 01-10-E 5단계 — 옵티마이저 클래스로 감싸기

## ⚠️ 참고

- **3. 01-10-E 5단계 — 옵티마이저 클래스로 감싸기** — 책의 옵티마이저 코드는 출력이 없어, 15걸음 후 위치를 확인하는 출력 코드를 **추가**했습니다(표시된 부분).

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 23_1_partial_derivative_gradient.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
