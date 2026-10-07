# #06 · 1부 미분이란? 자동차 속도계로 이해하는 미분·연쇄법칙·그래디언트 (1부)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/06-1-derivatives/06_1_derivatives.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #06 · 1부의 실습 코드입니다.
> 원문: 책 **00-04 「미분·경사하강법 — 안개 낀 산에서 내려오기 (1부)」**

미분을 "지금 이 순간의 변화 속도"로 이해하고, 전진차분과 중심차분으로 수치미분을 직접 구현해 오차를 비교합니다. 손으로 유도한 연쇄법칙·시그모이드 도함수를 수치미분으로 채점하고, 순전파/역전파 클래스와 편미분·그래디언트까지 코드로 확인합니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [00-04 미분·경사하강법 — 안개 낀 산에서 내려오기 (1부)](https://wikidocs.net/439840) |
| ✍️ 블로그 글 | [https://wikidocs.net/blog/@mldict/32650/](https://wikidocs.net/blog/@mldict/32650/) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`06_1_derivatives.py`](06_1_derivatives.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`06_1_derivatives.ipynb`](06_1_derivatives.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`) |

## 🧭 코드 순서

1. 1단계 — 미분이란? 전진차분 vs 중심차분
2. 2단계 — 미분 공식과 연쇄법칙 검산
3. 2단계 — 순전파·역전파 클래스 (Square, Sigmoid)
4. 3단계 — 편미분과 그래디언트 (수치 그래디언트)

## ⚠️ 참고

- **3. 2단계 — 순전파·역전파 클래스 (Square, Sigmoid)** — `Function` 기반 클래스는 책 04-03장에서 만드는 것이라 본문 코드만으로는 실행되지 않습니다. 실행을 위해 최소한의 `Function` 정의와 동작 확인 코드를 **추가**했습니다(표시된 부분).

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 06_1_derivatives.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
