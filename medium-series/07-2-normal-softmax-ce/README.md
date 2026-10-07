# #07 · 2부 정규분포·소프트맥스·교차 엔트로피 — AI는 확률로 대답한다 (2부)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/07-2-normal-softmax-ce/07_2_normal_softmax_ce.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #07 · 2부의 실습 코드입니다.
> 원문: 책 **00-05 「확률·베이즈 정리 — 병원 검사의 함정 (2부)」**

정규분포 PDF를 공식 그대로 구현해 SciPy와 비교하고, 68-95-99.7 규칙과 중심극한정리를 데이터로 검증합니다. 오버플로에 안전한 소프트맥스와 교차 엔트로피를 구현해 기울기 ŷ−y 를 수치미분으로 확인하고, 우도비를 누적하는 나이브 베이즈 스팸 필터를 만듭니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [00-05 확률·베이즈 정리 — 병원 검사의 함정 (2부)](https://wikidocs.net/439838) |
| ✍️ 블로그 글 | [https://wikidocs.net/blog/@mldict/32651/](https://wikidocs.net/blog/@mldict/32651/) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`07_2_normal_softmax_ce.py`](07_2_normal_softmax_ce.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`07_2_normal_softmax_ce.ipynb`](07_2_normal_softmax_ce.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`, `scipy`) |

## 🧭 코드 순서

1. 00-05-D · 4단계 — 정규분포와 68-95-99.7 규칙
2. 00-05-E · 5단계 — 소프트맥스와 교차 엔트로피
3. 🔴 심화 — 나이브 베이즈 스팸 필터

## ⚠️ 참고

- **3. 🔴 심화 — 나이브 베이즈 스팸 필터** — 책 본문에 실린 실행 결과(무료 당첨 → 99.4%, 회의 보고서 → 0.9%)는 이 코드를 실제로 실행한 값(98.8%, 0.8%)과 다릅니다. 사전 오즈 0.3/0.7 × 10 × 20 ≈ 85.7 → 확률 ≈ 98.8% 이므로 코드의 계산이 맞습니다.

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 07_2_normal_softmax_ce.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
