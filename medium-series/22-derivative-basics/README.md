# #22 미분 기초 — 계단·속도계로 이해하는 순간 변화율과 경사하강법

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/22-derivative-basics/22_derivative_basics.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #22의 실습 코드입니다.
> 원문: 책 **01-09 「미분 기초 — 순간 변화율과 경사」**

계단의 기울기(세로 ÷ 가로)에서 출발해 간격 h를 줄이며 순간 기울기로 수렴하는 과정을 숫자로 보고, 거듭제곱 미분 규칙과 직접 재보기 검산, 기울기를 따라 골짜기 바닥으로 내려가는 경사하강법과 보폭(학습률)의 효과를 코드로 확인합니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | [미분 기초 — 계단·속도계로 이해하는 순간 변화율과 경사하강법](https://medium.com/p/6dc119c4c961) |
| 📘 책 원문 페이지 | [01-09 미분 기초 — 순간 변화율과 경사](https://wikidocs.net/439748) |
| ✍️ 블로그 글 | [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/) (이 편의 블로그 글은 아직 없음) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`22_derivative_basics.py`](22_derivative_basics.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`22_derivative_basics.ipynb`](22_derivative_basics.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (없음 — 표준 파이썬만 사용) |

## 🧭 코드 순서

1. 1단계 — 기울기 = 세로 ÷ 가로
2. 2단계 — 간격 h를 줄이면 순간 기울기에 수렴
3. 3단계 — 거듭제곱 미분 규칙
4. 3단계 — 공식 vs 직접 재보기
5. 4단계 — 골짜기 바닥 찾아 내려가기
6. 4단계 — 보폭이 크면? 작으면?

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 22_derivative_basics.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
