# #24 · 2부 확률은 넓이다 — 적분으로 읽는 확률밀도·기댓값·KL 발산·ELBO (2부)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/24-2-integral-probability-expectation/24_2_integral_probability_expectation.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #24 · 2부의 실습 코드입니다.
> 원문: 책 **01-11 「적분 — 넓이와 확률의 수학 (2부)」**

f(x)=x² 를 [0,3]에서 조각 수 n = 10 → 10000 으로 늘려가며 왼쪽 리만 합과 사다리꼴 공식(np.trapezoid)의 오차를 비교하고, 표준정규분포의 구간 확률(CDF 차), 지수분포 기댓값의 수치적분, 몬테카를로 근사(= loss.mean() 의 원리)를 코드로 확인합니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [01-11 적분 — 넓이와 확률의 수학 (2부)](https://wikidocs.net/439757) |
| ✍️ 블로그 글 | [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/) (이 편의 블로그 글은 아직 없음) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`24_2_integral_probability_expectation.py`](24_2_integral_probability_expectation.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`24_2_integral_probability_expectation.ipynb`](24_2_integral_probability_expectation.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`, `scipy`) |

## 🧭 코드 순서

1. 01-11-J 9단계 — 리만 합과 사다리꼴 공식 비교
2. 01-11-J 9단계 — 확률과 기댓값을 적분으로 확인

## ⚠️ 참고

- **1. 01-11-J 9단계 — 리만 합과 사다리꼴 공식 비교** — `np.trapezoid` 는 NumPy 2.0 이상에서 쓸 수 있습니다(1.x 에서는 `np.trapz`). Colab 기본 환경에서 그대로 실행됩니다.
- **2. 01-11-J 9단계 — 확률과 기댓값을 적분으로 확인** — 책의 실행 결과에 붙은 `(이론값 1/λ = …)` 설명은 출력문이 아니라 해설이라 코드 실행 시에는 숫자만 출력됩니다.

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 24_2_integral_probability_expectation.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
