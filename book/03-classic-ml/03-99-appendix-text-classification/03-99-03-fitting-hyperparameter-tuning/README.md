# depth 후보를 하나씩 바꿔가며 학습/검증 정확도를 비교합니다

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-99-03 ⚖ 과소적합·과대적합과 하이퍼파라미터 튜닝 — 옷 맞춤 비유.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_99_03_fitting_hyperparameter_tuning.py`](03_99_03_fitting_hyperparameter_tuning.py) | 코드 블록 2개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_99_03_fitting_hyperparameter_tuning.ipynb`](03_99_03_fitting_hyperparameter_tuning.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 3단계 — 코드로 직접 확인하기 | 13 |
| 2 | 4단계 — 하이퍼파라미터 튜닝: GridSearchCV | 21 |

## 실행 방법

```bash
cd book/03-classic-ml/03-99-appendix-text-classification/03-99-03-fitting-hyperparameter-tuning
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_99_03_fitting_hyperparameter_tuning.py
```

또는 `03_99_03_fitting_hyperparameter_tuning.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: 앞 절 03-99-02 의 코드 실행 결과 (`Xtr` 등) — 책이 '앞 절에서 만든 것을 그대로 쓴다'고 가정
- 스크립트가 멈춘 지점: `NameError: name 'Xtr' is not defined. Did you mean: 'str'?`
- 블록별 실행(오류가 나도 다음 블록 계속): 0/2 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
