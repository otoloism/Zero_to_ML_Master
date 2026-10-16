# #24 · 1부 적분이란? 피자 조각으로 이해하는 리만 합·정적분·미적분학의 기본정리 (1부)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/24-1-integral-riemann-ftc/24_1_integral_riemann_ftc.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #24 · 1부의 실습 코드입니다.
> 원문: 책 **01-11 「적분 — 넓이와 확률의 수학 (1부)」**

시그마 Σ가 사실은 for 문이라는 것을 Σi² = 55 계산으로 확인하고, SymPy로 원시함수를 자동으로 찾은 뒤(∫3x²dx = x³, ∫₁⁴3x²dx = 63) 미분으로 검산합니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [01-11 적분 — 넓이와 확률의 수학 (1부)](https://wikidocs.net/439757) |
| ✍️ 블로그 글 | [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/) (이 편의 블로그 글은 아직 없음) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`24_1_integral_riemann_ftc.py`](24_1_integral_riemann_ftc.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`24_1_integral_riemann_ftc.ipynb`](24_1_integral_riemann_ftc.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`sympy`) |

## 🧭 코드 순서

1. 01-11-B 1단계 — 시그마 Σ를 for 문으로 번역하기
2. 01-11-G 6단계 — SymPy로 원시함수 찾고 미분으로 검산

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 24_1_integral_riemann_ftc.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
