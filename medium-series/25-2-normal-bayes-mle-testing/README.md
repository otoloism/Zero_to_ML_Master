# #25 · 2부 베이즈 정리와 최대가능도추정(MLE) — 정규분포부터 손실 함수의 뿌리, p값·신뢰구간까지 (2부)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otoloism/Zero_to_ML_Master/blob/main/medium-series/25-2-normal-bayes-mle-testing/25_2_normal_bayes_mle_testing.ipynb)

> **Zero to 머신러닝 딥러닝 Master** Medium 시리즈 #25 · 2부의 실습 코드입니다.
> 원문: 책 **01-12 「확률과 통계 기초 — 불확실성을 숫자로 (2부)」**

표준정규분포의 68–95–99.7 규칙을 cdf 로 확인하고, 동전 10번 중 앞면 7번에서 가능도를 최대로 만드는 p = 0.7(MLE)을 격자 탐색으로 찾고, 10-fold 교차검증 정확도로 대응표본 t-검정·p값·신뢰구간을 계산한 뒤, 중심·퍼짐·모양·이상치를 한 번에 보는 데이터 진단 함수를 만듭니다.

## 📎 링크

| | |
|---|---|
| 📝 Medium 글 | **TODO: Medium 게시 후 URL 입력** (`https://medium.com/p/...`) |
| 📘 책 원문 페이지 | [01-12 확률과 통계 기초 — 불확실성을 숫자로 (2부)](https://wikidocs.net/439756) |
| ✍️ 블로그 글 | [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/) (이 편의 블로그 글은 아직 없음) |

## 📂 파일

| 파일 | 설명 |
|---|---|
| [`25_2_normal_bayes_mle_testing.py`](25_2_normal_bayes_mle_testing.py) | 책의 코드 블록을 순서대로 모은 실행 스크립트 (섹션 주석 포함) |
| [`25_2_normal_bayes_mle_testing.ipynb`](25_2_normal_bayes_mle_testing.ipynb) | 같은 코드를 제목·설명과 함께 담은 Jupyter 노트북 (실행 결과 포함) |
| [`requirements.txt`](requirements.txt) | 필요한 패키지 (`numpy`, `scipy`) |

## 🧭 코드 순서

1. 01-12-H 8단계 — 정규분포 확률 계산
2. 01-12-J 10단계 — MLE 를 수치적으로 확인
3. 01-12-K 11단계 — t-검정과 신뢰구간
4. 01-12-L 12단계 — 데이터 진단 파이프라인
5. 연습문제 A9 — 분산 직접 구현

## ⚠️ 참고

- **2. 01-12-J 10단계 — MLE 를 수치적으로 확인** — 책 본문의 실행 결과 중 p=0.3·p=0.9 의 로그가능도(-9.4949 · -7.6461)는 실제 계산값(7·ln0.3+3·ln0.7 = -9.4978, 7·ln0.9+3·ln0.1 = -7.6453)과 소수 셋째 자리가 다릅니다. 코드는 그대로이며 이 노트북의 출력이 정확한 값입니다.
- **5. 연습문제 A9 — 분산 직접 구현** — 연습문제 해설 코드는 함수 정의만 있어, 4단계의 시험 점수로 np.var 와 비교하는 코드를 **추가**했습니다(표시된 부분).

## ▶️ 실행 방법

```bash
pip install -r requirements.txt
python 25_2_normal_bayes_mle_testing.py
```

또는 위의 **Open in Colab** 배지를 눌러 브라우저에서 바로 실행하세요 (설치 불필요).

## 🔗 함께 보기

- 📘 **책 (WikiDocs)** — [Zero to 머신러닝 딥러닝 Master](https://wikidocs.net/book/21464)
- 🏠 **홈페이지** — [mldict.net](https://mldict.net)
- ✍️ **기술 블로그** — [wikidocs.net/blog/@mldict](https://wikidocs.net/blog/@mldict/)
- 💻 **전체 코드 저장소** — [github.com/otoloism/Zero_to_ML_Master](https://github.com/otoloism/Zero_to_ML_Master)
