# #14 내적·외적·유사도 행렬 — 추천 시스템과 어텐션 QKᵀ의 수학

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/14-dot-cross-outer-similarity/14_dot_cross_outer_similarity.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #14의 실습 코드입니다.
> 원문: 책 **01-05 「행렬 내적·외적·유사도 행렬」**

내적의 계산식과 각도식이 같은 값을 주는지 확인하고, 크기 때문에 내적이 거짓말을 하는 예를 코사인으로 바로잡습니다. 벡터 외적(cross)과 외적(outer)을 결과의 형태로 구분하고, 유사도 행렬 XXᵀ·코사인 유사도 행렬로 추천을 굴린 뒤 셀프 어텐션 softmax(QKᵀ/√d)V 를 손으로 따라갑니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [01-05 행렬 내적·외적·유사도 행렬](https://wikidocs.net/439750) |
| ✍️ 블로그 글 | [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/) (이 편의 블로그 글은 아직 없음) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`14_dot_cross_outer_similarity.py`](14_dot_cross_outer_similarity.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`14_dot_cross_outer_similarity.ipynb`](14_dot_cross_outer_similarity.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`) |

## 🧭 코드 순서

1. 1단계 — 내적의 두 얼굴: 계산식 vs 각도식
2. 2단계 — 크기의 함정: 내적 vs 코사인
3. 3단계 — 벡터 외적(cross): 방향이 나온다
4. 4단계 — 외적(outer): 결과는 행렬
5. 5단계 — 유사도 행렬 XXᵀ
6. 6단계 — 코사인 유사도 행렬로 추천
7. 7단계 — 어텐션: softmax(QKᵀ/√d)V

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 14_dot_cross_outer_similarity.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
