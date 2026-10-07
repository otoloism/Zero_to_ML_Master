# #18 행렬식(Determinant) 쉽게 이해하기 — 넓이와 부피의 배율로 보는 det

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/18-determinant/18_determinant.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #18의 실습 코드입니다.
> 원문: 책 **01-06-02 「행렬식(Determinant) — 정사각행렬 속에 숨은 넓이와 부피」**

2×2·3×3·4×4 행렬식을 np.linalg.det 로 계산해 손계산(ad−bc, 사러스, 여인수 전개)과 비교하고, 행 교환·스칼라배·행 덧셈이 행렬식을 어떻게 바꾸는지, det(AB) = det(A)det(B) 를 검증합니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [01-06-02 행렬식(Determinant) — 정사각행렬 속에 숨은 넓이와 부피](https://wikidocs.net/439761) |
| ✍️ 블로그 글 | [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/) (이 편의 블로그 글은 아직 없음) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`18_determinant.py`](18_determinant.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`18_determinant.ipynb`](18_determinant.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`) |

## 🧭 코드 순서

1. 2단계 — 2×2 행렬식 ad − bc
2. 4단계 — 3×3 계산법 ① 사러스 법칙 검산
3. 5단계 — 3×3 계산법 ② 여인수 전개 검산 (4×4 포함)
4. 6단계 — 계산을 쉽게 만드는 행 연산 요령
5. 7단계 — det(AB) = det(A)det(B)

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 18_determinant.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
