# #10 머신러닝에 수학이 왜 필요할까? — 라면과 요리로 보는 AI 수학 공부법

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/10-why-math/10_why_math.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #10의 실습 코드입니다.
> 원문: 책 **01-01 「왜 수학인가? — 요리를 배울 것인가? 라면만 끓이다 갈 것인가!」**

파이썬·NumPy 환경이 준비됐는지 확인하는 첫 코드와, 00권 여섯 장(벡터·행렬·최소제곱·경사하강법·소프트맥스·SVD)을 20줄로 맛보는 코드를 실행합니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [01-01 왜 수학인가? — 요리를 배울 것인가? 라면만 끓이다 갈 것인가!](https://wikidocs.net/439746) |
| ✍️ 블로그 글 | [https://wikidocs.net/blog/@mldict/32654/](https://wikidocs.net/blog/@mldict/32654/) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`10_why_math.py`](10_why_math.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`10_why_math.ipynb`](10_why_math.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`) |

## 🧭 코드 순서

1. 01-01-D · 4단계 — 부엌 차리기: 환경 확인
2. 01-01-E · 5단계 — 맛보기: 여섯 장을 20줄로

## ⚠️ 참고

- **2. 01-01-E · 5단계 — 맛보기: 여섯 장을 20줄로** — 책 본문에서 `x = x - 0.1 * df(x)` 줄 끝의 `← 이 한 줄이 AI 학습의 전부!` 가 주석 기호(`#`) 없이 들어가 있어 SyntaxError 가 납니다. 여기서는 `# ←` 로 주석 처리해 실행되게 했습니다.
- **1단계 — "라이브러리 한 줄" 예시 (실행하지 않음)** — 라이브러리만 쓰면 학습이 한 줄로 끝난다는 것을 보여주는 예시입니다. `X_train` 등 데이터가 정의되어 있지 않은 설명용 코드라 **실행하지 않습니다**. (노트북에는 코드만 표시)

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 10_why_math.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
