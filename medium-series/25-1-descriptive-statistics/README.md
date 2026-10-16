# #25 · 1부 확률과 통계 기초 — 평균·분산·공분산으로 데이터의 중심·퍼짐·관계 읽기 (1부)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/25-1-descriptive-statistics/25_1_descriptive_statistics.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #25 · 1부의 실습 코드입니다.
> 원문: 책 **01-12 「확률과 통계 기초 — 불확실성을 숫자로 (1부)」**

시험 점수 5개로 평균·중앙값·모분산(ddof=0)·표본분산(ddof=1)·표준편차를 구해 직접 구현과 비교하고, 공부 시간과 점수의 공분산·상관계수·공분산 행렬, 왜도·첨도와 로그 변환 전처리, 소프트맥스와 크로스엔트로피 손실을 NumPy·SciPy로 확인합니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [01-12 확률과 통계 기초 — 불확실성을 숫자로 (1부)](https://wikidocs.net/439756) |
| ✍️ 블로그 글 | [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/) (이 편의 블로그 글은 아직 없음) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`25_1_descriptive_statistics.py`](25_1_descriptive_statistics.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`25_1_descriptive_statistics.ipynb`](25_1_descriptive_statistics.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`, `scipy`) |

## 🧭 코드 순서

1. 01-12-D 4단계 — 평균·분산·표준편차 (ddof)
2. 01-12-E 5단계 — 공분산 · 상관계수 · 공분산 행렬
3. 01-12-F 6단계 — 왜도·첨도 측정과 전처리
4. 01-12-G 7단계 — 소프트맥스 직접 구현

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 25_1_descriptive_statistics.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
