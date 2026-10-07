# #17 연립방정식 풀이법 — 대입법과 그래프 교점으로 잇는 대수와 기하

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/17-solving-systems/17_solving_systems.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #17의 실습 코드입니다.
> 원문: 책 **01-06-01 「연립방정식의 의미와 풀이법 — 대수와 기하를 잇는 다리」**

SymPy로 1차-2차 연립방정식을 풀고, 원과 직선의 교점 개수(0·1·2개)로 해의 개수를 기하적으로 해석한 뒤, 2차-2차 연립방정식을 인수분해와 solve 로 검산합니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [01-06-01 연립방정식의 의미와 풀이법 — 대수와 기하를 잇는 다리](https://wikidocs.net/439759) |
| ✍️ 블로그 글 | [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/) (이 편의 블로그 글은 아직 없음) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`17_solving_systems.py`](17_solving_systems.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`17_solving_systems.ipynb`](17_solving_systems.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`sympy`) |

## 🧭 코드 순서

1. 2단계 — 1차-2차 연립방정식: 대입법 검산
2. 3단계 — 해의 개수 ↔ 그래프의 교점
3. 4단계 — 2차-2차 연립방정식: 인수분해

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 17_solving_systems.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
