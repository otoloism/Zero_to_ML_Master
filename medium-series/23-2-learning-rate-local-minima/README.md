# #23 · 2부 경사하강법의 함정 — 학습률·국소 최솟값·신경망 학습이 같은 원리인 이유 (2부)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/23-2-learning-rate-local-minima/23_2_learning_rate_local_minima.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #23 · 2부의 실습 코드입니다.
> 원문: 책 **01-10 「편미분·그래디언트·경사하강법 (2부)」**

학습률 0.01 / 0.1 / 0.25 / 0.5 / 0.9 로 30걸음씩 걸어 수렴·정확 도착·진동·발산을 비교하고, 웅덩이가 두 개인 f(x) = x⁴ − 4x² + 0.5x 에서 시작점에 따라 전역/지역 최솟값으로 갈리는 것을 확인한 뒤, 연습문제의 일반화 그래디언트와 수치 그래디언트를 구현합니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | [Medium에서 읽기](https://medium.com/p/c1910d65c827) |
| 📘 책 원문 페이지 | [01-10 편미분·그래디언트·경사하강법 (2부)](https://wikidocs.net/439754) |
| ✍️ 블로그 글 | [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/) (이 편의 블로그 글은 아직 없음) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`23_2_learning_rate_local_minima.py`](23_2_learning_rate_local_minima.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`23_2_learning_rate_local_minima.ipynb`](23_2_learning_rate_local_minima.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`) |

## 🧭 코드 순서

1. 01-10-F 6단계 — 학습률 5종 비교 실험
2. 01-10-G 7단계 — 시작점에 따라 갈리는 운명 (지역 최솟값)
3. 연습문제 — f(x, y) = a·x² + b·y² 의 그래디언트
4. 연습문제 — 수치 그래디언트

## ⚠️ 참고

- **1. 01-10-F 6단계 — 학습률 5종 비교 실험** — 이 코드는 1부(5단계)의 `compute_loss`·`compute_gradient` 를 사용하므로, 그 정의를 앞에 **추가**했습니다(표시된 부분).
- **4. 연습문제 — 수치 그래디언트** — 연습문제 해설 코드는 함수 정의만 있어, 실습 1(점 (2, 1))로 공식과 수치 그래디언트를 비교하는 코드를 **추가**했습니다(표시된 부분).
- **01-10-H 8단계 — PyTorch 학습 루프의 세 줄** — optimizer.zero_grad() → loss.backward() → optimizer.step(). 모델·손실이 정의되어 있어야 하는 설명용 코드입니다. (노트북에는 코드만 표시)

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 23_2_learning_rate_local_minima.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
