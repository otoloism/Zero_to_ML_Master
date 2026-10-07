# 서로 다른 알고리즘 세 개를 준비합니다 (서로 다른 "위원"들)

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-99-04 🤝 앙상블 모델, 불균형 데이터, 상관 계수 — 협업과 균형.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_99_04_ensemble_imbalance_correlation.py`](03_99_04_ensemble_imbalance_correlation.py) | 코드 블록 4개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_99_04_ensemble_imbalance_correlation.ipynb`](03_99_04_ensemble_imbalance_correlation.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 시각화 | 18 |
| 2 | 시각화 | 15 |
| 3 | 시각화 | 13 |
| 4 | 예: 0.42 → 양의 상관 (해당 단어가 있으면 긍정일 가능성이 높음) | 6 |

## 실행 방법

```bash
cd book/03-classic-ml/03-99-appendix-text-classification/03-99-04-ensemble-imbalance-correlation
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_99_04_ensemble_imbalance_correlation.py
```

또는 `03_99_04_ensemble_imbalance_correlation.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **🧩 문맥 필요(조각)** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: 앞 절 03-99-02 의 코드 실행 결과 (`Xtr` 등) — 책이 '앞 절에서 만든 것을 그대로 쓴다'고 가정
- 스크립트가 멈춘 지점: `NameError: name 'Xtr' is not defined. Did you mean: 'str'?`
- 블록별 실행(오류가 나도 다음 블록 계속): 1/4 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
