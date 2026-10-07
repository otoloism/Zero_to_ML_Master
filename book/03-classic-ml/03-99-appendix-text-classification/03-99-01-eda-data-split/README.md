# positive 800

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-99-01🔍 데이터 탐색과 데이터 분할 — 요리 전 재료 점검하기.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_99_01_eda_data_split.py`](03_99_01_eda_data_split.py) | 코드 블록 2개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_99_01_eda_data_split.ipynb`](03_99_01_eda_data_split.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 📖 처음부터 끝까지 상세 풀이 | 12 |
| 2 | ④ 코드 구현 | 16 |

## 실행 방법

```bash
cd book/03-classic-ml/03-99-appendix-text-classification/03-99-01-eda-data-split
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_99_01_eda_data_split.py
```

또는 `03_99_01_eda_data_split.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **⏭️ 외부 요구사항** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: 외부 데이터 파일 `movie_reviews.csv` (책에 데이터 생성 코드 없음 — 직접 내려받아 같은 폴더에 두세요)
- 스크립트가 멈춘 지점: `FileNotFoundError: [Errno 2] No such file or directory: 'movie_reviews.csv'`
- 블록별 실행(오류가 나도 다음 블록 계속): 0/2 블록 성공

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
