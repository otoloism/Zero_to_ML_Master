# #19 역행렬(Inverse Matrix) 구하는 법 — 가우스-조던 소거법부터 (AB)⁻¹ = B⁻¹A⁻¹까지

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/19-inverse-matrix/19_inverse_matrix.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #19의 실습 코드입니다.
> 원문: 책 **01-06-03 「역행렬(Inverse Matrix) — 행렬의 나눗셈을 완성하다」**

2×2 역행렬을 NumPy로 구해 AA⁻¹ = I 를 검산하고, SymPy rref 로 가우스-조던 소거법을 재현합니다. 행렬식·수반행렬 공식을 직접 구현하고, 역행렬이 없는 특이행렬과 (AB)⁻¹ = B⁻¹A⁻¹ 성질을 확인합니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [01-06-03 역행렬(Inverse Matrix) — 행렬의 나눗셈을 완성하다](https://wikidocs.net/439760) |
| ✍️ 블로그 글 | [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/) (이 편의 블로그 글은 아직 없음) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`19_inverse_matrix.py`](19_inverse_matrix.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`19_inverse_matrix.ipynb`](19_inverse_matrix.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`, `sympy`) |

## 🧭 코드 순서

1. 3단계 — 예제① 2×2 역행렬과 검산
2. 4단계 — 예제② 3×3 역행렬: 가우스-조던 (SymPy rref)
3. 5단계 — 행렬식·수반행렬 공식 (2×2 전용)
4. 6단계 — 역행렬이 존재하지 않는 경우
5. 7단계 — 역행렬의 성질 (AB)⁻¹ = B⁻¹A⁻¹

## ⚠️ 참고

- **4. 6단계 — 역행렬이 존재하지 않는 경우** — det(C) 는 실행 환경(NumPy/LAPACK)에 따라 `0.0` 또는 `6.66e-16` 같은 0에 아주 가까운 값으로 출력될 수 있습니다(부동소수점 오차).

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 19_inverse_matrix.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
