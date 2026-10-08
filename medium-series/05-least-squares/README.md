# #05 최소제곱법과 정규방정식 — 완벽한 답이 없을 때 최선의 답 찾기

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/05-least-squares/05_least_squares.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #05의 실습 코드입니다.
> 원문: 책 **00-03 「연립방정식 — "x가 뭔지 맞춰보세요" 게임」**

식이 미지수보다 많은 과결정계에서 "오차 제곱합이 가장 작은 답"을 찾는 최소제곱법을 다룹니다. 정규방정식 x̂ = (AᵀA)⁻¹Aᵀb 를 그대로 구현한 방법과 실무 표준 `np.linalg.lstsq` 를 키·몸무게 데이터로 비교하고, 잔차·오차제곱합·직교 조건(Aᵀe = 0)까지 검증합니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | [최소제곱법과 정규방정식 — 완벽한 답이 없을 때 최선의 답 찾기](https://medium.com/p/933746f6a6a1) |
| 📘 책 원문 페이지 | [00-03 연립방정식 — "x가 뭔지 맞춰보세요" 게임](https://wikidocs.net/439841) |
| ✍️ 블로그 글 | [https://wikidocs.net/blog/@mldict/32649/](https://wikidocs.net/blog/@mldict/32649/) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`05_least_squares.py`](05_least_squares.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`05_least_squares.ipynb`](05_least_squares.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`) |

## 🧭 코드 순서

1. 00-03-R3 🔴 구현 원리와 코드 — 공식·lstsq 두 방법 비교

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 05_least_squares.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
