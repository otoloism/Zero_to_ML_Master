# 02-01 🌱 1부 — 파이썬이라는 언어의 이야기

📖 출처: 「Zero to 머신러닝 딥러닝 Master」 [WikiDocs](https://wikidocs.net/book/21464) · 원고 파일 `02-01 🌱 1부 — 파이썬이라는 언어의 이야기.md`

## 파일

| 파일 | 설명 |
|---|---|
| [`02_01_python_story.py`](02_01_python_story.py) | 코드 블록 1개를 순서대로 이어 붙인 실행 스크립트 (`# %% [Block n]` 구분) |
| [`02_01_python_story.ipynb`](02_01_python_story.ipynb) | 같은 코드를 블록당 셀 1개로 담은 주피터 노트북 |
| [`commands.sh`](commands.sh) | 책에 나온 셸 명령(설치 등) |

## 코드 블록 목록

| # | 내용 | 줄 수 |
|---|---|---|
| 1 | 💡 가상환경(venv) — 프로젝트마다 방을 따로 쓰기 | 5 |

## 실행 방법

```bash
cd book/02-hello-python/02-01-python-story
pip install -r ../../requirements.txt   # 처음 한 번 (book/requirements.txt)
python 02_01_python_story.py
```

또는 `02_01_python_story.ipynb` 를 Jupyter / VS Code / Colab 에서 열어 셀을 순서대로 실행하세요.

## 검증 결과

- 상태: **✅ 실행 OK** (CPU, 제한 시간 120초, `MPLBACKEND=Agg`)

## 셸 명령

```bash
# 파이썬이 잘 설치됐는지 확인
python --version

# 이 장에서 쓸 라이브러리 설치 (pip = 파이썬 패키지 설치 도구)
pip install numpy plotly

# 주피터 노트북까지 원한다면
pip install jupyterlab

# 설치된 목록 확인
pip list
```

---

책 원문 코드를 그대로 옮겼습니다. 출력 결과·수식·의사코드 블록은 제외했습니다.
