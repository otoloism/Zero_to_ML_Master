# 03-09-03 그레이디언트 부스팅 — 잔차를 학습하는 릴레이

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-09-03 그레이디언트 부스팅 — 잔차를 학습하는 릴레이.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_09_03_gradient_boosting.py`](03_09_03_gradient_boosting.py) | 코드 블록 7개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_09_03_gradient_boosting.ipynb`](03_09_03_gradient_boosting.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 2단계 — 강의 자료 7페이지 재현 — 트리 3그루 — 강의 자료 7페이지 재현 — 잔차를 이어 학습하는 트리 3그루 | 39 |
| 2 | 사이킷런과 완전히 같은가? | 14 |
| 3 | 직관 — 손실 함수를 바꾸면 '잔차'도 바뀐다 — 유도한 식이 진짜인지 컴퓨터로 검산합니다. | 36 |
| 4 | 초기값 \\(F_0\\)는 왜 평균인가? — 사이킷런의 GradientBoosting 이 정말 'y의 평균'에서 시작하는지 확인합니다. | 23 |
| 5 | 5단계 — 회귀 모델과 분류 모델 — 두 형제 — 회귀와 분류를 나란히 실행해 봅니다. | 40 |
| 6 | 실습 — [실습 1 정답 예시] 이상치가 하나 섞였을 때 두 손실 함수의 반응 | 21 |
| 7 | 실습 — [실습 2 정답 예시] 트리 개수별 시험 RMSE — 최적 지점 찾기 | 20 |

## 실행 방법

```bash
cd book/03-classic-ml/03-09-boosting/03-09-03-gradient-boosting
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_09_03_gradient_boosting.py
```

또는 `03_09_03_gradient_boosting.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

## 추출 시 수정 사항

- 이중 줄바꿈(추출 아티팩트) 정리

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
