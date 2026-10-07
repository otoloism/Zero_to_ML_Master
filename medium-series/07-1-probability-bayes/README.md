# #07 · 1부 베이즈 정리 쉽게 이해하기 — 병원 검사의 함정 (1부: 확률·조건부 확률)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/07-1-probability-bayes/07_1_probability_bayes.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #07 · 1부의 실습 코드입니다.
> 원문: 책 **00-05 「확률·베이즈 정리 — 병원 검사의 함정 (1부)」**

P(A)=n(A)/n(Ω) 와 대수의 법칙을 주사위 시뮬레이션으로 확인하고, 분할표 데이터에서 불리언 마스크로 조건부 확률을 계산합니다. 베이즈 정리로 "양성인데 실제로 병일 확률"을 구하고, 발병률·오탐률·재검사에 따라 사후 확률이 어떻게 바뀌는지 탐색합니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [00-05 확률·베이즈 정리 — 병원 검사의 함정 (1부)](https://wikidocs.net/439838) |
| ✍️ 블로그 글 | [https://wikidocs.net/blog/@mldict/32651/](https://wikidocs.net/blog/@mldict/32651/) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`07_1_probability_bayes.py`](07_1_probability_bayes.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`07_1_probability_bayes.ipynb`](07_1_probability_bayes.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`) |

## 🧭 코드 순서

1. 00-05-A · 1단계 — 확률이란? (이론값과 대수의 법칙)
2. 00-05-B · 2단계 — 조건부 확률과 독립
3. 00-05-C · 3단계 — 베이즈 정리: 병원 검사의 함정

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 07_1_probability_bayes.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
