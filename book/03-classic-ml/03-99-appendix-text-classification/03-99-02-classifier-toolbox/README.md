# kernel='linear': 직선(평면) 경계 사용 (텍스트처럼 고차원엔 보통 이걸로 충분)

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-99-02 🧰 일반적인 머신러닝 모델 한눈에 — 분류기 도구함.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_99_02_classifier_toolbox.py`](03_99_02_classifier_toolbox.py) | 코드 블록 6개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_99_02_classifier_toolbox.ipynb`](03_99_02_classifier_toolbox.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 03-99-02-A 1) 로지스틱 회귀 (Logistic Regression) — 로지스틱 회귀 모델 생성 및 학습 | 6 |
| 2 | 03-99-02-B 2) 결정 트리 (Decision Tree) | 6 |
| 3 | 03-99-02-C 3) 랜덤 포레스트 (Random Forest) | 6 |
| 4 | 03-99-02-D 4) SVM (Support Vector Machine) | 5 |
| 5 | 03-99-02-E 5) 나이브 베이즈 (Naive Bayes) | 4 |
| 6 | alpha: 라플라스 스무딩. 학습 데이터에 없던 단어가 나와도 확률이 0이 되지 않게 함 | 25 |

## 실행 방법

```bash
cd book/03-classic-ml/03-99-appendix-text-classification/03-99-02-classifier-toolbox
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_99_02_classifier_toolbox.py
```

또는 `03_99_02_classifier_toolbox.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 상세: `LogisticRegression` 이(가) 이 절의 뒤쪽 블록에서야 정의됩니다 (책 서술 순서상 앞 블록은 설명용 조각).
- 스크립트가 멈춘 지점: `NameError: name 'LogisticRegression' is not defined`
- 블록별 실행(오류가 나도 다음 블록 계속): 0/6 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
