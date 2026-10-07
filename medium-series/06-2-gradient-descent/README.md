# #06 · 2부 경사하강법 완전 정복 — 안개 낀 산에서 가장 낮은 곳 찾기 (2부)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/06-2-gradient-descent/06_2_gradient_descent.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #06 · 2부의 실습 코드입니다.
> 원문: 책 **00-04 「미분·경사하강법 — 안개 낀 산에서 내려오기 (2부)」**

경사하강법의 업데이트 식 x ← x − η·f′(x) 를 한 줄 코드로 구현하고, 학습률에 따른 수렴·진동·발산 조건(|1−2η| < 1)을 실험으로 확인합니다. 2변수 그래디언트 하강, 선형회귀 학습(정규방정식 답과 비교), 그리고 PyTorch·TensorFlow·JAX 자동미분으로 같은 결과를 재현합니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [00-04 미분·경사하강법 — 안개 낀 산에서 내려오기 (2부)](https://wikidocs.net/439840) |
| ✍️ 블로그 글 | [https://wikidocs.net/blog/@mldict/32650/](https://wikidocs.net/blog/@mldict/32650/) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`06_2_gradient_descent.py`](06_2_gradient_descent.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`06_2_gradient_descent.ipynb`](06_2_gradient_descent.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`, `torch`, `tensorflow`, `jax`) |

## 🧭 코드 순서

1. 5단계 — 경사하강법 구현과 이론 예측
2. 5단계 — 학습률 실험: 수렴·진동·발산
3. 5단계 — 2변수 경사하강법
4. 5단계 — 선형회귀를 경사하강법으로 학습
5. 6단계 — 프레임워크 연결: PyTorch·TensorFlow·JAX 자동미분

## ⚠️ 참고

- **5. 6단계 — 프레임워크 연결: PyTorch·TensorFlow·JAX 자동미분** — `torch`, `tensorflow`, `jax` 가 필요합니다(설치 용량이 큽니다). Colab에는 기본 설치되어 있습니다. TensorFlow/JAX 는 첫 import 시 몇 초가 걸리고, GPU가 없으면 CPU 관련 안내 로그가 출력될 수 있습니다(정상). 세 프레임워크 모두 기본이 float32 라서 결과가 책의 2.009904 대신 2.009903 처럼 마지막 자리가 다를 수 있습니다.

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 06_2_gradient_descent.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
