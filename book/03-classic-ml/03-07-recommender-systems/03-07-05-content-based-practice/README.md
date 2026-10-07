# 03-07-05 콘텐츠 기반 추천 시스템 실습 (Content Based Recommendation with TMDB 5000)

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `03-07-05 콘텐츠 기반 추천 시스템 실습.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`03_07_05_content_based_practice.py`](03_07_05_content_based_practice.py) | 코드 블록 17개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`03_07_05_content_based_practice.ipynb`](03_07_05_content_based_practice.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 1단원 — 실습 환경 세팅 | 8 |
| 2 | 1단원 — 실습 환경 세팅 | 10 |
| 3 | 2단원 — 데이터 불러오기와 주요 컬럼 추출 — 사용 데이터: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metada… | 4 |
| 4 | 2단원 — 데이터 불러오기와 주요 컬럼 추출 — 주요 컬럼으로 데이터 프레임 생성 | 4 |
| 5 | 3단원 — 전처리 : 문자열을 파이썬 객체로 되살리기 | 2 |
| 6 | 3단원 — 전처리 : 문자열을 파이썬 객체로 되살리기 | 3 |
| 7 | 3단원 — 전처리 : 문자열을 파이썬 객체로 되살리기 | 13 |
| 8 | 4단원 — CountVectorizer로 장르를 숫자로 | 15 |
| 9 | 🧪 직접 돌려보기 — 5편짜리 축소판 | 25 |
| 10 | 공식 | 5 |
| 11 | 공식 | 8 |
| 12 | 6단원 — 유사 영화 추천 함수 | 21 |
| 13 | 6단원 — 유사 영화 추천 함수 | 10 |
| 14 | 직관 — 저울의 눈금 | 15 |
| 15 | 직관 — 저울의 눈금 | 14 |
| 16 | 🚀 확장 — 유사도와 가중 평점을 결합한 최종 추천 함수 | 24 |
| 17 | 실습 — 실습 3 정답 — m을 바꿔가며 '무명 만점 영화'가 어떻게 되는지 관찰 | 21 |

## 실행 방법

```bash
cd book/03-classic-ml/03-07-recommender-systems/03-07-05-content-based-practice
pip install -r ../../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 03_07_05_content_based_practice.py
```

또는 `03_07_05_content_based_practice.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **⏭️ 외부 요구사항** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)
- 필요 사항: 외부 데이터 파일 `tmdb_5000_movies.csv` (책에 데이터 생성 코드 없음 — 직접 내려받아 같은 폴더에 두세요)
- 스크립트가 멈춘 지점: `FileNotFoundError: [Errno 2] No such file or directory: 'tmdb_5000_movies.csv'`
- 블록별 실행(오류가 나도 다음 블록 계속): 11/17 블록 성공

## 추출 시 수정 사항

- 주피터 전용 명령(`%matplotlib`, `!pip` 등)은 .py 에서 주석 처리했습니다 (노트북에는 원문 그대로).

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
