# -*- coding: utf-8 -*-
"""
02-08 🐼 8부 — Pandas 파이썬으로 쓰는 엑셀

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 02권 🐍 헬로 파이썬 — 파이썬의 탄생부터 NumPy · Pandas  · Plotly까지/02-08 🐼 8부 — Pandas 파이썬으로 쓰는 엑셀.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 🧒 시작하기 전에 — 머신러닝을 공부하는데 왜 판다스부터 배울까? — 사고 ① 숫자인데 '글자'로 저장된 키
# 사고 ① 숫자인데 '글자'로 저장된 키
heights = ["165", "158"]                  # 파일에서 읽으면 이렇게 글자로 들어오는 일이 흔합니다
print("키의 합:", heights[0] + heights[1])  # 더하기를 했는데...?


# %% [Block 2] 🧒 시작하기 전에 — 머신러닝을 공부하는데 왜 판다스부터 배울까? — 사고 ② 이름과 점수를 따로따로 리스트에 담았더니...
# 사고 ② 이름과 점수를 따로따로 리스트에 담았더니...
names  = ["민준", "서연", "도윤", "하은"]
scores = [88, 92, 75]                     # 도윤의 점수를 깜빡하고 안 적음 (한 칸이 빠짐)

for name, score in zip(names, scores):    # 짝지어 출력해 보면
    print(name, score)


# %% [Block 3] 🧒 시작하기 전에 — 머신러닝을 공부하는데 왜 판다스부터 배울까? — 사고 ③ 빈칸과 중복 제출이 섞인 설문
# 사고 ③ 빈칸과 중복 제출이 섞인 설문
scores2 = [88, 92, None, 92, 75]          # None = 빈칸, 92가 두 번 = 서연이 두 번 제출
valid = [m for m in scores2 if m is not None]
print("평균(중복 포함):", sum(valid) / len(valid))
print("올바른 평균    :", (88 + 92 + 75) / 3)


# %% [Block 4] 🧒 시작하기 전에 — 머신러닝을 공부하는데 왜 판다스부터 배울까?
import pandas as pd

# 설문 원본: 사람이 대충 적어서 지저분한 상태
raw = pd.DataFrame({
    "이름":   ["민준", "서연", "도윤", "서연", "하은"],
    "반":     ["1반", "2반", "1반", "2반", "2반"],
    "키":     ["165", "158", "172", "158", " 161 "],   # 글자 + 공백 섞임
    "몸무게": [55, None, 63, None, 50],                # 빈칸
    "수학":   [88, 92, None, 92, 75],                  # 빈칸 + 중복
})
print(raw)
print("-" * 40)
print(raw.dtypes)            # 키가 '숫자'가 아니라 '글자(str)' 칸이라는 걸 바로 알려 줌


# %% [Block 5] 🧒 시작하기 전에 — 머신러닝을 공부하는데 왜 판다스부터 배울까?
clean = raw.drop_duplicates()                                  # 사고 ③: 중복 제출 지우기
clean["키"] = clean["키"].str.strip().astype(int)              # 사고 ①: 공백 지우고 숫자로
clean["몸무게"] = clean["몸무게"].fillna(clean["몸무게"].mean()) # 빈 몸무게는 평균으로 채우기
clean = clean.dropna(subset=["수학"])                          # 정답(수학)이 없는 줄은 공부에 못 씀

print(clean)
print("-" * 40)
print("반별 수학 평균:")
print(clean.groupby("반")["수학"].mean())                      # 엑셀의 피벗 한 번


# %% [Block 6] 🧒 시작하기 전에 — 머신러닝을 공부하는데 왜 판다스부터 배울까?
X = clean[["키", "몸무게"]]      # 📘 문제지: 모델이 보고 판단할 정보
y = clean["수학"]                # 📗 정답지: 맞혀야 할 값

print("X (문제지) 모양:", X.shape)
print(X)
print("y (정답지):", y.tolist())


# %% [Block 7] 02-08-AE 판다스 시작하기 — 파이썬 안의 엑셀 켜기 🟢
import pandas as pd            # 판다스를 pd라는 짧은 별명으로 불러옵니다
import numpy as np             # 판다스는 NumPy 위에서 동작하므로 함께 불러 두면 편합니다

print("판다스 버전:", pd.__version__)   # 설치된 판다스 버전 확인
print("NumPy 버전:", np.__version__)


# %% [Block 8] 02-08-AF Series — 이름표 붙은 한 줄 🟢
import pandas as pd

# 값 4개에 이름표 a~d를 붙인 Series 만들기
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print(s)
print("-" * 30)
print("index :", s.index)      # 이름표 목록
print("values:", s.values)     # 값만 꺼낸 NumPy 배열
print("dtype :", s.dtype)      # 값의 자료형


# %% [Block 9] 02-08-AF Series — 이름표 붙은 한 줄 🟢
print(s['c'])          # 이름표 하나로 꺼내면 → 값 하나(스칼라)
print("-" * 30)
print(s[['b', 'c']])   # 이름표 목록(리스트)으로 꺼내면 → Series
print("-" * 30)
print(s[s > 2])        # 조건으로 거르기 → 조건이 True인 칸만 남은 Series


# %% [Block 10] 02-08-AF Series — 이름표 붙은 한 줄 🟢
s2 = pd.Series([1, 2, 3, 4])   # index를 주지 않으면?
print(s2.index)                # 0부터 시작하는 RangeIndex가 자동으로 붙음

# 딕셔너리로 만들면 key가 곧 이름표가 됩니다
population = pd.Series({'서울': 940, '부산': 330, '대구': 237})
print(population)
print("서울 인구(만 명):", population['서울'])


# %% [Block 11] 02-08-AF Series — 이름표 붙은 한 줄 🟢
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([10, 20, 30], index=['b', 'c', 'd'])

print(s1 + s2)          # 같은 이름표끼리 더하기 → a, d는 짝이 없어 NaN
print("-" * 30)
print(s1.add(s2, fill_value=0))   # 짝이 없을 때 0으로 간주하고 더하기
print("-" * 30)
print(s1 * 10)          # 스칼라 연산은 모든 칸에 한꺼번에 (브로드캐스트)


# %% [Block 12] 02-08-AG DataFrame 만들기 — 엑셀 시트 한 장 🟢
import pandas as pd

# 🅰️ 딕셔너리로 만들기 — key는 열 이름, value는 그 열의 값 목록
data_dict = {
    '이름': ['소라', '창훈', '영미'],
    '나이': [25, 30, 35],
    '거주지': ['인천', '순천', '동해']
}
df_dict = pd.DataFrame(data_dict)
print(df_dict)


# %% [Block 13] 02-08-AG DataFrame 만들기 — 엑셀 시트 한 장 🟢 — 🅱️ 2중 리스트로 만들기 — 안쪽 리스트 하나가 한 행
# 🅱️ 2중 리스트로 만들기 — 안쪽 리스트 하나가 한 행
data_list = [
    ['소라', 25, '인천'],
    ['창훈', 30, '순천'],
    ['영미', 35, '동해']
]
columns = ['이름', '나이', '거주지']           # 열 제목은 따로 알려 줘야 함
df_list = pd.DataFrame(data_list, columns=columns)
print(df_list)
print("두 표가 똑같은가?", df_dict.equals(df_list))


# %% [Block 14] 02-08-AG DataFrame 만들기 — 엑셀 시트 한 장 🟢
print("index  :", df_list.index)      # 행 이름표 (RangeIndex)
print("columns:", df_list.columns)    # 열 이름표 (Index 객체)
print("values :")
print(df_list.values)                 # 2차원 NumPy 배열
print("dtypes :")
print(df_list.dtypes)                 # 열마다 자료형


# %% [Block 15] 02-08-AG DataFrame 만들기 — 엑셀 시트 한 장 🟢
age = df_list['나이']          # 열 하나를 꺼내면?
print(type(age))               # → Series
print(age)

# 행 이름표를 직접 지정할 수도 있습니다
df_named = pd.DataFrame(data_dict, index=['A01', 'A02', 'A03'])
print(df_named)


# %% [Block 16] 02-08-AH 파일 읽고 쓰기 — CSV와 엑셀 🟢
import pandas as pd

data = {
    '연도': [2010, 2011, 2012, 2013, 2014],
    '인구(만명)': [5000, 5050, 5100, 5150, 5200],
    'GDP(억달러)': [1000, 1050, 1100, 1150, 1200]
}
df_years = pd.DataFrame(data)

df_years.to_csv('years_data.csv')          # 기본값: index도 함께 저장, utf-8

loaded = pd.read_csv('years_data.csv')     # 다시 읽어 보면?
print(loaded)


# %% [Block 17] 02-08-AH 파일 읽고 쓰기 — CSV와 엑셀 🟢 — 해결 ① 저장할 때 index를 빼기
# 해결 ① 저장할 때 index를 빼기
df_years.to_csv('years_data_noidx.csv', index=False)
print(pd.read_csv('years_data_noidx.csv'))
print("-" * 40)

# 해결 ② 읽을 때 첫 번째 열을 index로 쓰기
print(pd.read_csv('years_data.csv', index_col=0))


# %% [Block 18] 02-08-AH 파일 읽고 쓰기 — CSV와 엑셀 🟢 — 윈도우 엑셀에서 한글이 안 깨지게 cp949로 저장
# 윈도우 엑셀에서 한글이 안 깨지게 cp949로 저장
df_years.to_csv('years_data_cp949.csv', encoding='cp949', index=False)

# 같은 파일을 잘못된 암호표(utf-8)로 열면?
try:
    pd.read_csv('years_data_cp949.csv')                  # 기본 encoding='utf-8'
except UnicodeDecodeError as e:
    print("❌ 에러:", type(e).__name__)

# 올바른 암호표로 열기
print(pd.read_csv('years_data_cp949.csv', encoding='cp949'))


# %% [Block 19] 02-08-AH 파일 읽고 쓰기 — CSV와 엑셀 🟢 — read_csv의 자주 쓰는 옵션들
# read_csv의 자주 쓰는 옵션들
print(pd.read_csv('years_data_noidx.csv', usecols=['연도', 'GDP(억달러)']))  # 필요한 열만
print("-" * 40)
print(pd.read_csv('years_data_noidx.csv', nrows=2))                          # 앞 2행만
print("-" * 40)
print(pd.read_csv('years_data_noidx.csv', skiprows=[1, 2]))                  # 제목줄(0번) 다음 1·2번 줄(2010·2011년) 건너뛰기
print("-" * 40)
print(pd.read_csv('years_data_noidx.csv', header=0,
                  names=['year', 'pop', 'gdp']))                              # 열 이름 새로 붙이기


# %% [Block 20] 02-08-AH 파일 읽고 쓰기 — CSV와 엑셀 🟢
books_df = pd.DataFrame({
    '도서ID': [1, 2, 3, 4],
    '도서명': ['파이썬 프로그래밍', '데이터 분석', '머신러닝 입문', '인공지능의 이해'],
    '저자': ['김파이썬', '이데이터', '박머신', '최인공'],
    '출판년도': [2019, 2020, 2018, 2021]
})
members_df = pd.DataFrame({
    '회원ID': [101, 102, 103],
    '이름': ['홍길동', '김영희', '이철수'],
    '가입년도': [2020, 2021, 2019]
})
borrow_df = pd.DataFrame({
    '회원ID': [101, 102, 101, 103],
    '도서ID': [2, 3, 1, 4],
    '대출일': ['2024-03-01', '2024-03-02', '2024-03-05', '2024-03-07']
})

books_df.to_excel('도서_데이터.xlsx', index=False)     # 시트 하나짜리 엑셀 파일
print(pd.read_excel('도서_데이터.xlsx'))


# %% [Block 21] 02-08-AH 파일 읽고 쓰기 — CSV와 엑셀 🟢 — 여러 DataFrame을 한 파일의 여러 시트에 저장
# 여러 DataFrame을 한 파일의 여러 시트에 저장
with pd.ExcelWriter('도서_데이터_전체.xlsx') as writer:
    books_df.to_excel(writer, sheet_name='도서_목록', index=False)
    members_df.to_excel(writer, sheet_name='회원_목록', index=False)
    borrow_df.to_excel(writer, sheet_name='대출_기록', index=False)

# sheet_name=None → 모든 시트를 {시트이름: DataFrame} 딕셔너리로 읽기
sheets = pd.read_excel('도서_데이터_전체.xlsx', sheet_name=None)
print("시트 목록:", list(sheets.keys()))
print(sheets['대출_기록'])
print("-" * 40)
print(pd.read_excel('도서_데이터_전체.xlsx', sheet_name='회원_목록'))   # 시트 하나만


# %% [Block 22] 02-08-AH 파일 읽고 쓰기 — CSV와 엑셀 🟢 — 한 시트 안에 표 여러 개를 위치를 정해 배치하기 (startrow, startcol은 0부터)
# 한 시트 안에 표 여러 개를 위치를 정해 배치하기 (startrow, startcol은 0부터)
with pd.ExcelWriter('도서_데이터_onesheet.xlsx') as writer:
    books_df.to_excel(writer, sheet_name='도서데이터', index=False)
    members_df.to_excel(writer, sheet_name='도서데이터', startrow=0, startcol=5, index=False)
    borrow_df.to_excel(writer, sheet_name='도서데이터', startrow=6, startcol=0, index=False)

raw = pd.read_excel('도서_데이터_onesheet.xlsx', header=None)   # 시트를 있는 그대로 보기
print(raw.fillna(''))


# %% [Block 23] 02-08-AI 데이터 첫인상 — shape · info · describe · head 🟢
import pandas as pd

data = {
    "이름": ["김철수", "이영희", "박민수"],
    "나이": [25, 32, 19],
    "도시": ["서울", "부산", "대구"]
}
df = pd.DataFrame(data)
print(df)
print("데이터프레임의 차원:", df.shape)     # (행 수, 열 수) 튜플
print("행 수:", df.shape[0], "/ 열 수:", df.shape[1])
print(df.columns)


# %% [Block 24] 02-08-AI 데이터 첫인상 — shape · info · describe · head 🟢
df_copy = df.copy()                       # 원본을 지키기 위해 복사본에서 실험
df_copy.columns = ['A', 'B', 'C']         # 열 이름 전체를 한 번에 바꾸기 (개수가 같아야 함)
print(df_copy)
print("-" * 30)
print(df.rename(columns={'도시': '거주도시'}))   # 일부만 바꿀 때는 rename


# %% [Block 25] 02-08-AI 데이터 첫인상 — shape · info · describe · head 🟢
df.info()       # 문진표: 행 수, 열마다 결측 아닌 개수, 자료형, 메모리


# %% [Block 26] 02-08-AI 데이터 첫인상 — shape · info · describe · head 🟢
print(df.describe())                     # 숫자 열의 요약 통계
print("-" * 30)
print(df.describe(include='all'))        # 문자열 열까지 포함 (unique, top, freq)


# %% [Block 27] 02-08-AI 데이터 첫인상 — shape · info · describe · head 🟢
print(df.head(1))     # 앞에서 1행 (기본값 5행)
print("-" * 30)
print(df.tail(1))     # 뒤에서 1행 (기본값 5행)
print("-" * 30)
# 출력 설정을 잠깐만 바꾸기: 최대 2행, 2열까지만 표시
with pd.option_context('display.max_rows', 2, 'display.max_columns', 2):
    print(df)


# %% [Block 28] 02-08-AJ 골라 보기 — [] · loc · iloc 🟢 ⭐
column_data = df['도시']               # 열 하나 → Series
print(type(column_data).__name__)
print(column_data)
print("-" * 30)
selected_columns = df[['도시', '이름']]  # 열 여러 개(순서도 내 마음대로) → DataFrame
print(selected_columns)
print("-" * 30)
result = df[['도시']]                   # 열 하나지만 표 모양으로 받고 싶으면 이중 대괄호
print(type(result).__name__)
print(result)


# %% [Block 29] 02-08-AJ 골라 보기 — [] · loc · iloc 🟢 ⭐
print(df.loc[0])                       # 0번 행 → Series (열 이름이 이름표가 됨)
print("-" * 30)
print(df.loc[[0, 2]])                  # 떨어진 행 여러 개 → DataFrame
print("-" * 30)
print(df.loc[1:2])                     # 이어진 행: 1부터 2까지 (끝 포함!)


# %% [Block 30] 02-08-AJ 골라 보기 — [] · loc · iloc 🟢 ⭐
print(df.loc[:, '도시'])                # 모든 행(:)의 도시 열 → Series
print("-" * 30)
print(df.loc[:, ['도시', '이름']])       # 모든 행, 열 두 개 → DataFrame
print("-" * 30)
print(df.loc[:, '나이':'도시'])          # 나이부터 도시까지 (끝 포함)
print("-" * 30)
print(df.loc[0, '이름'])                # 셀 하나 → 값(스칼라)
print("-" * 30)
print(df.loc[1:2, ['이름', '나이']])     # 행 범위 × 열 목록


# %% [Block 31] 02-08-AJ 골라 보기 — [] · loc · iloc 🟢 ⭐
print(df.iloc[0])          # 첫 번째 행
print("-" * 30)
print(df.iloc[:, 1])       # 두 번째 열(나이)
print("-" * 30)
print(df.iloc[-1])         # 마지막 행 (음수 = 뒤에서부터)
print("-" * 30)
print(df.iloc[:, -2])      # 뒤에서 두 번째 열


# %% [Block 32] 02-08-AJ 골라 보기 — [] · loc · iloc 🟢 ⭐
print(df.iloc[0:2])        # 0, 1번째 행 (2는 미포함!)
print("-" * 30)
print(df.iloc[:, 1:2])     # 1번째 열만 (2는 미포함) → DataFrame
print("-" * 30)
print(df.iloc[[0, 2]])     # 떨어진 행
print("-" * 30)
print(df.iloc[:, [0, 2]])  # 떨어진 열
print("-" * 30)
print(df[0:2])             # 대괄호에 슬라이스를 넣으면 행을 위치로 자름 (끝 미포함)


# %% [Block 33] 02-08-AJ 골라 보기 — [] · loc · iloc 🟢 ⭐ — loc과 iloc이 갈리는 순간: 이름표가 위치와 다를 때
# loc과 iloc이 갈리는 순간: 이름표가 위치와 다를 때
df_idx = df.set_index('이름')            # 이름 열을 행 이름표로
print(df_idx)
print("-" * 30)
print(df_idx.loc['이영희'])               # 이름표로
print("-" * 30)
print(df_idx.iloc[1])                    # 위치로 — 같은 행
print("-" * 30)
print(df_idx.loc['김철수':'이영희', '나이'])  # 이름표 범위도 끝 포함


# %% [Block 34] 02-08-AK 조건 필터 — 엑셀의 자동 필터 🟢 ⭐
mask = df['나이'] > 20          # ① 마스크 만들기
print(mask)
print("-" * 30)
filtered_rows1 = df[mask]       # ② 마스크 적용 (한 줄로 df[df['나이'] > 20] 도 같음)
print(filtered_rows1)


# %% [Block 35] 02-08-AK 조건 필터 — 엑셀의 자동 필터 🟢 ⭐ — 조건 여러 개: 각 조건을 반드시 괄호로 감싸기
# 조건 여러 개: 각 조건을 반드시 괄호로 감싸기
filtered_rows2 = df[(df['나이'] > 20) & (df['도시'] != '서울')]
print(filtered_rows2)
print("-" * 30)
print(df[(df['나이'] < 20) | (df['도시'] == '서울')])   # 또는
print("-" * 30)
print(df[~(df['도시'] == '서울')])                      # 아니다 (서울이 아닌 사람)


# %% [Block 36] 02-08-AK 조건 필터 — 엑셀의 자동 필터 🟢 ⭐
filtered_rows_isin = df[df['도시'].isin(['대구', '부산'])]   # 목록 안에 있는 값이면 True
print(filtered_rows_isin)
print("-" * 30)
print(df.loc[df['나이'] >= 25, ['이름', '나이']])           # 조건 + 열 고르기는 loc으로 한 번에
print("-" * 30)
print(df.query("나이 > 20 and 도시 != '서울'"))             # 문자열로 조건 쓰기 (SQL 느낌)


# %% [Block 37] 02-08-AK 조건 필터 — 엑셀의 자동 필터 🟢 ⭐ — 괄호를 빼먹으면 어떻게 될까?
# 괄호를 빼먹으면 어떻게 될까?
try:
    df[df['나이'] > 20 & df['도시'] != '서울']
except Exception as e:
    print("❌", type(e).__name__)


# %% [Block 38] 02-08-AL 행·열 추가·삭제·값 수정 — 표 고치기 🔵
import pandas as pd

data = {
    "주문번호": [1, 2, 3, 4, 5, 6, 7],
    "고객명": ["유재석", "조세호", "유재석", "권상우", "조세호", "권상우", "유재석"],
    "주문금액": [20000, 50000, 15000, 30000, 100000, 25000, 20000],
    "주문일자": ["2024-03-01", "2024-03-01", "2024-03-02", "2024-03-02",
                 "2024-03-03", "2024-03-03", "2024-03-04"],
    "배송지역": ["서울", "부산", "서울", "대구", "부산", "대구", "서울"]
}
df = pd.DataFrame(data)

df['배송료'] = 3000                     # 맨 오른쪽에 새 열 (모든 행에 3000)
df.insert(6, '배송상태', '배송완료')     # 위치 6(7번째 자리)에 새 열 끼우기
print(df)


# %% [Block 39] 02-08-AL 행·열 추가·삭제·값 수정 — 표 고치기 🔵
new_order = {"주문번호": 8, "고객명": "이무진", "주문금액": 45000, "주문일자": "2024-03-05"}
df.loc[7] = new_order          # 이름표 7인 행이 없으므로 새로 만들어짐
print(df.tail(2))              # 딕셔너리에 없던 열은 NaN(빈 칸)


# %% [Block 40] 02-08-AL 행·열 추가·삭제·값 수정 — 표 고치기 🔵 — concat: 표와 표를 이어 붙이기 — 딕셔너리를 한 줄짜리 DataFrame으로 만들어 붙임
# concat: 표와 표를 이어 붙이기 — 딕셔너리를 한 줄짜리 DataFrame으로 만들어 붙임
new_dict = {'주문번호': 9, '고객명': '이상순', '주문금액': 40000, '배송료': 3000}
df = pd.concat([df, pd.DataFrame([new_dict])], ignore_index=True)

new_dict = {'주문번호': 10, '고객명': '이효리', '주문금액': 35000, '배송료': 3000}
df = pd.concat([df, pd.DataFrame([new_dict])], ignore_index=True)
print(df.tail(3))
print("-" * 40)

# 여러 행을 한꺼번에 — 결과를 새 변수에 (원본 df는 그대로 10행 유지)
new_orders_df = pd.DataFrame({'주문번호': [11, 12],
                              '고객명': ['김종국', '하하'],
                              '주문금액': [25000, 15000],
                              '배송료': [3000, 3000]})
df_updated = pd.concat([df, new_orders_df], ignore_index=True)
print("df:", df.shape, "/ df_updated:", df_updated.shape)


# %% [Block 41] 02-08-AL 행·열 추가·삭제·값 수정 — 표 고치기 🔵 — Series를 행으로 바꿔 붙이는 방법도 있지만…
# Series를 행으로 바꿔 붙이는 방법도 있지만…
new_series = pd.Series([9, '이상순', 40000, 3000], index=['주문번호', '고객명', '주문금액', '배송료'])
tmp = pd.concat([df.head(2), new_series.to_frame().T], ignore_index=True)
print(tmp.dtypes)
print("-" * 30)
print(tmp.infer_objects().dtypes)     # 자료형을 다시 추론해 복구


# %% [Block 42] 02-08-AL 행·열 추가·삭제·값 수정 — 표 고치기 🔵
concat_series = pd.Series(
    [1000, 2000, 1500, 3000, 5000, 2500, 2000, 1000, 4000],   # 9개뿐 (df는 10행)
    name='할인금액'                                           # name이 열 이름이 됨
)
df = pd.concat([df, concat_series], axis=1, join='inner')    # 공통 index(0~8)만 남김
print(df)


# %% [Block 43] 02-08-AL 행·열 추가·삭제·값 수정 — 표 고치기 🔵
print(df.drop('배송료', axis=1).head(3))                 # 열 하나 삭제 (axis=1 = 열)
print("-" * 40)
print(df.drop(['주문일자', '배송지역'], axis=1).head(3))   # 열 여러 개
print("-" * 40)
print(df.drop(8, axis=0).tail(3))                       # 이름표 8인 행 삭제 (axis=0 = 행)
print("-" * 40)
print("원본은 그대로:", df.shape)                         # drop은 원본을 바꾸지 않음


# %% [Block 44] 02-08-AL 행·열 추가·삭제·값 수정 — 표 고치기 🔵
df.iloc[0, 5] = 2500                                   # 0번째 행, 5번째 열(배송료)
df.loc[1, '배송료'] = df.loc[1, '배송료'] - 500          # 기존 값에서 500 빼기
print(df.loc[0:1, ['고객명', '배송료']])
print("-" * 40)
df.loc[df['고객명'] != '유재석', '배송료'] = 2800         # 조건에 맞는 행들의 배송료를 한 번에
print(df[['고객명', '배송료']])


# %% [Block 45] 02-08-AM 결측치 — 빈 칸 찾고, 채우고, 지우기 🔵 ⭐
print(df.isnull())                 # 칸마다 비었으면 True


# %% [Block 46] 02-08-AM 결측치 — 빈 칸 찾고, 채우고, 지우기 🔵 ⭐
missing_values_count = df.isnull().sum()          # 열마다 True 개수 = 결측치 수
print(missing_values_count)
print("-" * 30)
missing_values_count2 = df.isnull().sum(axis=1)   # 행마다 결측치 수
print(missing_values_count2)
print("-" * 30)
print("결측 비율(%):")
print((df.isnull().mean() * 100).round(1))        # True=1, False=0 → 평균 = 비율
print("-" * 30)
non_missing_values = df[~df['배송지역'].isnull()] # 배송지역이 있는 행만 (= notnull)
print(non_missing_values[['고객명', '배송지역']])


# %% [Block 47] 02-08-AM 결측치 — 빈 칸 찾고, 채우고, 지우기 🔵 ⭐
df_filled = df.fillna('확인중')          # 모든 빈 칸을 한 값으로
print(df_filled.tail(2))
print("-" * 40)
print(df.fillna({'배송상태': '배송중', '배송지역': '미정'}).tail(2))   # 열마다 다른 값


# %% [Block 48] 02-08-AM 결측치 — 빈 칸 찾고, 채우고, 지우기 🔵 ⭐ — ❌ 예전 책·블로그에 흔한 코드 (판다스 3.0부터 원본이 바뀌지 않음)
# ❌ 예전 책·블로그에 흔한 코드 (판다스 3.0부터 원본이 바뀌지 않음)
df_test = df.copy()
df_test['배송지역'].fillna(df_test['배송지역'].mode()[0], inplace=True)
print("❌ 연쇄 inplace 후 결측:", df_test['배송지역'].isnull().sum())

# ✅ 올바른 방법: 결과를 열에 다시 대입
print("최빈값:", df['배송지역'].mode()[0])
df['배송지역'] = df['배송지역'].fillna(df['배송지역'].mode()[0])
df['배송상태'] = df['배송상태'].fillna('배송중')
print("✅ 다시 대입 후 결측:", df['배송지역'].isnull().sum())
print(df[['고객명', '배송지역', '배송상태']].tail(3))


# %% [Block 49] 02-08-AM 결측치 — 빈 칸 찾고, 채우고, 지우기 🔵 ⭐ — ffill/bfill은 순서가 의미 있는 데이터(시간순)에서 빛납니다
# ffill/bfill은 순서가 의미 있는 데이터(시간순)에서 빛납니다
temp = pd.DataFrame({
    '시각': ['09:00', '10:00', '11:00', '12:00', '13:00'],
    '기온': [18.0, None, None, 21.0, None]       # 센서가 중간에 꺼짐
})
temp['ffill'] = temp['기온'].ffill()             # 직전 값으로 (채우기 핸들 ↓)
temp['bfill'] = temp['기온'].bfill()             # 다음 값으로 ↑ (마지막 칸은 다음 값이 없어 NaN)
temp['보간'] = temp['기온'].interpolate()        # 앞뒤 값 사이를 직선으로 이어 채우기
print(temp)


# %% [Block 50] 02-08-AM 결측치 — 빈 칸 찾고, 채우고, 지우기 🔵 ⭐
df['주문일자'] = df['주문일자'].ffill()          # 주문일자가 빈 이상순 → 직전 주문일로
print(df[['고객명', '주문일자']].tail(3))
print("-" * 30)
print("남은 결측치 합계:", df.isnull().sum().sum())   # 0이면 깨끗


# %% [Block 51] 02-08-AM 결측치 — 빈 칸 찾고, 채우고, 지우기 🔵 ⭐
survey = pd.DataFrame({
    '이름': ['가', '나', '다', '라'],
    '나이': [20, None, 35, None],
    '키':   [170, 165, None, None],
    '비고': [None, None, None, None]        # 아무도 안 적은 열
})
print(survey)
print("-" * 30)
print(survey.dropna(axis=1, how='all'))          # 전부 빈 열만 삭제 → 비고 삭제
print("-" * 30)
print(survey.dropna(axis=0, subset=['나이']))      # 나이가 빈 행만 삭제
print("-" * 30)
print(survey.dropna(thresh=3))                    # 값이 3개 이상 있는 행만 남김
print("-" * 30)
print(survey.dropna())                            # 빈 칸이 하나라도 있으면 삭제 → 전부 사라짐!


# %% [Block 52] 02-08-AN 정렬 · 인덱스 · 중복 — 줄 세우고, 이름표 바꾸고, 겹친 것 없애기 🔵
df_sorting = df.sort_values(by='주문일자', ascending=False)        # 최신 주문부터
print(df_sorting[['주문번호', '고객명', '주문일자']].head(4))
print("-" * 40)
# 고객명은 가나다순, 같은 고객 안에서는 주문금액 큰 순
print(df.sort_values(by=['고객명', '주문금액'], ascending=[True, False])[['고객명', '주문금액']])
print("-" * 40)
df_sorting_columns = df.sort_index(axis=1)                         # 열 이름을 가나다순으로
print(list(df_sorting_columns.columns))


# %% [Block 53] 02-08-AN 정렬 · 인덱스 · 중복 — 줄 세우고, 이름표 바꾸고, 겹친 것 없애기 🔵
df = df.set_index('주문번호')            # 주문번호 열 → 행 이름표
print(df.head(3))
print("-" * 40)
print(df.loc[5, ['고객명', '주문금액']])   # 이제 주문번호 5번으로 바로 찾기
print("-" * 40)
df = df.reset_index()                    # 이름표를 다시 열로, 0부터 새 번호
print(df.head(3))


# %% [Block 54] 02-08-AN 정렬 · 인덱스 · 중복 — 줄 세우고, 이름표 바꾸고, 겹친 것 없애기 🔵 — 정렬·필터 뒤 뒤죽박죽된 이름표를 0부터 다시: drop=True면 옛 이름표는 버림
# 정렬·필터 뒤 뒤죽박죽된 이름표를 0부터 다시: drop=True면 옛 이름표는 버림
tmp = df[df['주문금액'] >= 30000]
print(tmp[['고객명', '주문금액']])
print("-" * 30)
print(tmp.reset_index(drop=True)[['고객명', '주문금액']])


# %% [Block 55] 02-08-AN 정렬 · 인덱스 · 중복 — 줄 세우고, 이름표 바꾸고, 겹친 것 없애기 🔵
print(df.duplicated().sum(), "개 — 모든 열이 같은 완전 중복 행")
print("-" * 30)
print(df.duplicated(subset='고객명'))      # 고객명이 앞에서 이미 나왔으면 True


# %% [Block 56] 02-08-AN 정렬 · 인덱스 · 중복 — 줄 세우고, 이름표 바꾸고, 겹친 것 없애기 🔵
first = df.drop_duplicates(subset='고객명')               # 고객별 첫 주문만 (keep='first' 기본)
print(first[['고객명', '주문일자', '주문금액']])
print("-" * 40)
last = df.drop_duplicates(subset='고객명', keep='last')   # 고객별 마지막 주문만
print(last[['고객명', '주문일자', '주문금액']])


# %% [Block 57] 02-08-AO 연산과 통계 — 표 전체를 한 번에 계산하기 🔵
df_number = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [10, 20, 30],
    'C': [2.5, 4.5, 1.5]
})
print(df_number / 2)               # 모든 칸을 2로 나누기
print("-" * 30)
print(df_number.add(10))           # 모든 칸에 10 더하기 (= df_number + 10)
print("-" * 30)
print(df_number['A'] + df_number['B'])   # 열끼리 계산 (엑셀의 =A1+B1을 끝까지)


# %% [Block 58] 02-08-AO 연산과 통계 — 표 전체를 한 번에 계산하기 🔵
df_another = pd.DataFrame({
    'B': [100, 200, 300],
    'C': [25, 45, 15],
    'D': [5, 10, 15]
})
print(df_number.mul(df_another))                 # 공통 열 B, C만 곱해지고 A, D는 NaN
print("-" * 40)
print(df_number.mul(df_another, fill_value=1))   # 짝이 없으면 1로 간주


# %% [Block 59] 02-08-AO 연산과 통계 — 표 전체를 한 번에 계산하기 🔵
math_score = {
    '이름': ['김철수', '박영희', '이민정', '최재원', '한지민'],
    '학년': ['3학년', '1학년', '1학년', '2학년', '3학년'],
    '점수': [88, 75, 68, 90, 83]
}
df_score = pd.DataFrame(math_score)
sc = df_score['점수']
print("합계:", sc.sum(), "/ 개수:", sc.count(), "/ 평균:", sc.mean(), "/ 중앙값:", sc.median())
print("최소:", sc.min(), "/ 최대:", sc.max(), "/ 최대 위치:", sc.idxmax())
print("분산:", round(sc.var(), 2), "/ 표준편차:", round(sc.std(), 2))
print("최빈 학년 후보:", df_score['학년'].mode().tolist())   # 1학년·3학년이 2명씩 동점 → 둘 다 반환
print("최빈 학년 [0]:", df_score['학년'].mode()[0])        # 동점이면 정렬상 첫 번째
print("-" * 40)
print("3학년 최고점:", df_score[df_score['학년'] == '3학년']['점수'].max())


# %% [Block 60] 02-08-AO 연산과 통계 — 표 전체를 한 번에 계산하기 🔵
import numpy as np
arr = np.array([88, 75, 68, 90, 83])
print("NumPy  np.std :", round(np.std(arr), 4), "  ← n으로 나눔 (ddof=0)")
print("판다스 .std() :", round(sc.std(), 4), "  ← n-1로 나눔 (ddof=1)")
print("판다스 ddof=0 :", round(sc.std(ddof=0), 4))


# %% [Block 61] 02-08-AO 연산과 통계 — 표 전체를 한 번에 계산하기 🔵
sales = pd.Series([100, 200, 150, 300], index=['1월', '2월', '3월', '4월'])
print(pd.DataFrame({
    '월매출': sales,
    '누적매출': sales.cumsum(),             # 누적 합
    '전월대비(%)': (sales.pct_change() * 100).round(1),   # 변화율
}))
print("-" * 40)
growth = pd.Series([1.10, 1.20, 0.90])     # 매달 10% 증가, 20% 증가, 10% 감소
print(growth.cumprod())                    # 누적 곱 → 원금 대비 배율


# %% [Block 62] 02-08-AP 문자열 처리 — .str로 찾기·바꾸기·나누기 🔵
data = {
    '이름': ['김 철수', '박 영희', '이 민정', '최 재원', '한 지민'],
    '직업': ['개발자', '요리사', '마케터', '작가', '의사']
}
df_str = pd.DataFrame(data)

df_str['이름_민'] = df_str['이름'].str.contains('민')            # '민'이 들어 있나?
df_str['직업_사'] = df_str['직업'].str.endswith('사')            # '사'로 끝나나?
df_str['성'] = df_str['이름'].str.split().str.get(0)             # 공백으로 나눠 첫 조각
df_str['직업+이름'] = df_str['직업'].str.cat(df_str['이름'], sep=' ')   # 두 열 이어 붙이기
df_str['이름_공백제거'] = df_str['이름'].str.replace(' ', '')      # 공백 없애기
df_str['이름_마지막'] = df_str['이름'].str[3]                     # 네 번째 글자(위치 3)
print(df_str)


# %% [Block 63] 02-08-AP 문자열 처리 — .str로 찾기·바꾸기·나누기 🔵
messy = pd.Series(['  Apple ', 'banana', 'CHERRY  ', 'apple pie'])
print(pd.DataFrame({
    '원본': messy,
    'strip': messy.str.strip(),                 # 앞뒤 공백 제거
    'lower': messy.str.strip().str.lower(),     # 소문자로 통일
    'title': messy.str.strip().str.title(),     # 단어 첫 글자만 대문자
    '길이': messy.str.len(),                     # 글자 수 (공백 포함)
    'a시작': messy.str.strip().str.lower().str.startswith('a'),
}))
print("-" * 40)
# 실무 단골: 대소문자·공백이 제각각인 값을 통일한 뒤 세기
print(messy.str.strip().str.lower().str.split().str[0].value_counts())


# %% [Block 64] 02-08-AP 문자열 처리 — .str로 찾기·바꾸기·나누기 🔵
phones = pd.Series(['010-1234-5678', '010 9876 5432', '01055556666'])
digits = phones.str.replace(r'[^0-9]', '', regex=True)       # 숫자가 아닌 것을 모두 삭제
print(digits)
print("-" * 30)
print(digits.str[:3] + '-' + digits.str[3:7] + '-' + digits.str[7:])   # 형식 통일


# %% [Block 65] 02-08-AQ 날짜 · 그룹 · 피벗 · 병합 — SUMIF·피벗 테이블·VLOOKUP을 코드로 🔵 ⭐
df['주문일자'] = pd.to_datetime(df['주문일자'])     # 글자 → 날짜(datetime64)
print(df['주문일자'].dtype)
print("-" * 40)
print(pd.DataFrame({
    '주문일자': df['주문일자'],
    '연': df['주문일자'].dt.year,            # .dt = 날짜 열 전용 도구함
    '월': df['주문일자'].dt.month,
    '일': df['주문일자'].dt.day,
    '요일': df['주문일자'].dt.day_name(),
    '연월': df['주문일자'].dt.to_period('M'),   # 월 단위 기간 (2024-03)
}).head(4))
print("-" * 40)
print("첫 주문~마지막 주문:", (df['주문일자'].max() - df['주문일자'].min()).days, "일")


# %% [Block 66] 02-08-AQ 날짜 · 그룹 · 피벗 · 병합 — SUMIF·피벗 테이블·VLOOKUP을 코드로 🔵 ⭐
print(df.groupby('고객명')['주문금액'].sum())            # 고객별 주문금액 합계 (SUMIF)
print("-" * 40)
print(df_score.groupby('학년')['점수'].max())             # 11단계의 3학년 최고점을 모든 학년에 한 번에
print("-" * 40)
summary_tbl = df.groupby('고객명')['주문금액'].agg(['count', 'sum', 'mean'])  # 여러 통계 한 번에
print(summary_tbl.sort_values('sum', ascending=False))


# %% [Block 67] 02-08-AQ 날짜 · 그룹 · 피벗 · 병합 — SUMIF·피벗 테이블·VLOOKUP을 코드로 🔵 ⭐ — 이름 붙인 집계: 결과 열 이름 = (원본 열, 함수)
# 이름 붙인 집계: 결과 열 이름 = (원본 열, 함수)
report = df.groupby('배송지역').agg(
    주문수=('주문번호', 'count'),
    총매출=('주문금액', 'sum'),
    평균할인=('할인금액', 'mean'),
)
print(report)
print("-" * 40)
# 그룹마다 마지막 한 줄: 고객별 가장 최근 주문 (15단계에서 '연도별 마지막 달'로 다시 씀)
print(df.sort_values('주문일자').groupby('고객명').tail(1)[['고객명', '주문일자', '주문금액']])


# %% [Block 68] 02-08-AQ 날짜 · 그룹 · 피벗 · 병합 — SUMIF·피벗 테이블·VLOOKUP을 코드로 🔵 ⭐
pv = pd.pivot_table(df,
                    index='배송지역',        # 행 영역
                    columns='고객명',        # 열 영역
                    values='주문금액',       # 값 영역
                    aggfunc='sum',          # 합계
                    fill_value=0,           # 빈 칸은 0
                    margins=True,           # 총합계 행·열 추가
                    margins_name='합계')
print(pv)


# %% [Block 69] 02-08-AQ 날짜 · 그룹 · 피벗 · 병합 — SUMIF·피벗 테이블·VLOOKUP을 코드로 🔵 ⭐ — 4단계에서 만든 도서관 표 세 개를 다시 사용
# 4단계에서 만든 도서관 표 세 개를 다시 사용
# 대출 기록에는 회원ID·도서ID만 있음 → 이름과 도서명을 다른 표에서 찾아오기
borrow_full = (borrow_df
               .merge(members_df[['회원ID', '이름']], on='회원ID', how='left')
               .merge(books_df[['도서ID', '도서명']], on='도서ID', how='left'))
print(borrow_full)


# %% [Block 70] 02-08-AQ 날짜 · 그룹 · 피벗 · 병합 — SUMIF·피벗 테이블·VLOOKUP을 코드로 🔵 ⭐
grade = pd.DataFrame({'고객명': ['유재석', '조세호', '권상우', '김종국'],
                      '등급': ['VIP', '골드', '실버', '골드']})
print(df[['고객명', '주문금액']].merge(grade, on='고객명', how='left').tail(4))   # 등급 없는 고객은 NaN
print("-" * 40)
print(df[['고객명', '주문금액']].merge(grade, on='고객명', how='inner').shape, "← inner: 양쪽에 다 있는 고객만")


# %% [Block 71] 02-08-AQ 날짜 · 그룹 · 피벗 · 병합 — SUMIF·피벗 테이블·VLOOKUP을 코드로 🔵 ⭐
df['배송료'] = df['배송료'].astype('int64')             # 2800.0 → 2800 (결측이 없어야 가능)
df = df.rename(columns={'주문금액': '결제금액'})         # 열 이름 바꾸기
print(df.dtypes[['배송료', '결제금액']])
print("-" * 40)
small = pd.DataFrame({'지역': ['서울', '부산'], '2022': [10, 5], '2023': [14, 8]})
print(small)
print("-" * 40)
print(small.set_index('지역').T)       # 행과 열 뒤집기 (엑셀: 선택하여 붙여넣기 → 행/열 바꿈)


# %% [Block 72] 02-08-AR 시각화 — 판다스 표를 Plotly 그래프로 🔴
import pandas as pd
import plotly.express as px

df_graph = pd.DataFrame({
    'x': [1, 2, 3, 4],
    'x제곱': [1, 4, 9, 16],
    '2배': [2, 4, 6, 8],
})
fig = px.line(df_graph, x='x', y=['x제곱', '2배'],           # y에 열 목록 → 선 여러 개 + 범례 자동
              title='기본 선그래프', markers=True,
              labels={'value': 'y값', 'variable': '그래프'})   # 축·범례 이름 바꾸기
fig.show()      # 코랩/주피터에서 그래프 표시


# %% [Block 73] 02-08-AR 시각화 — 판다스 표를 Plotly 그래프로 🔴
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4], y=[1, 4, 9, 16], name='그래프1',
    mode='lines+markers',                                   # 선 + 점
    line=dict(color='red', dash='dash', width=1.5),          # 빨간 점선, 굵기 1.5
    marker=dict(symbol='circle', size=9),                   # 동그라미 점
))
fig.update_layout(title='꾸민 선그래프', xaxis_title='x축', yaxis_title='y축',
                  template='ggplot2')                        # 스타일 테마 (plt.style.use 역할)
fig.update_xaxes(range=[0, 5], showgrid=True)               # x축 범위 + 격자
fig.update_yaxes(range=[0, 20], showgrid=True)
fig.show()


# %% [Block 74] 02-08-AR 시각화 — 판다스 표를 Plotly 그래프로 🔴
import plotly.io as pio
print(list(pio.templates))      # 사용할 수 있는 테마 이름들
pio.templates.default = 'plotly_white'   # 이후 모든 그래프의 기본 테마 변경


# %% [Block 75] 02-08-AR 시각화 — 판다스 표를 Plotly 그래프로 🔴
from plotly.subplots import make_subplots

x1, y1 = list(range(1, 10)), [2, 3, 4, 5, 6, 7, 8, 9, 10]
x2, y2 = list(range(1, 8)), [10, 9, 8, 7, 6, 5, 4]
x3, y3 = list(range(3, 12)), [5] * 9

fig = make_subplots(rows=1, cols=3, subplot_titles=['서브플롯 1', '서브플롯 2', '서브플롯 3'])
fig.add_trace(go.Scatter(x=x1, y=y1, line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=x2, y=y2, line=dict(color='green', dash='dash')), row=1, col=2)
fig.add_trace(go.Scatter(x=x3, y=y3, line=dict(color='blue', dash='dash')), row=1, col=3)
fig.update_layout(showlegend=False, height=350)
fig.show()


# %% [Block 76] 02-08-AR 시각화 — 판다스 표를 Plotly 그래프로 🔴
import numpy as np
rng = np.random.default_rng(0)
pts = pd.DataFrame({'키': rng.normal(170, 8, 100), '몸무게': rng.normal(65, 10, 100)})

px.scatter(pts, x='키', y='몸무게', opacity=0.6, title='산점도').show()      # opacity = alpha
px.histogram(pts, x='키', nbins=15, title='키 분포').show()                  # nbins = bins

region = pd.DataFrame({'지역': ['서울', '부산', '대구'], '매출': [450, 230, 120]})
px.bar(region, x='지역', y='매출', color='지역', title='지역별 매출').show()

fig = px.pie(region, names='지역', values='매출', title='매출 비중')
fig.update_traces(textinfo='percent+label',        # autopct 역할: 비율 + 이름 표시
                  pull=[0.1, 0, 0])                # explode 역할: 서울 조각만 10% 떼어 내기
fig.show()


# %% [Block 77] 02-08-AR 시각화 — 판다스 표를 Plotly 그래프로 🔴
pd.options.plotting.backend = 'plotly'          # df.plot()이 Plotly로 그려지도록 전환
fig = df_graph.plot(x='x', y='x제곱', kind='line', title='판다스 데이터프레임 선 그래프')
fig.update_layout(xaxis_title='X축', yaxis_title='Y축')
fig.show()
pd.options.plotting.backend = 'matplotlib'      # 원래대로 돌리기


# %% [Block 78] 02-08-AR 시각화 — 판다스 표를 Plotly 그래프로 🔴
fig.write_html('my_figure.html')                # 인터랙티브 그대로 HTML 파일로 (추가 설치 없음)
# 이미지로 저장하려면: pip install -U kaleido 후
# fig.write_image('my_figure.png', scale=3)     # scale=3 ≈ dpi 300 효과


# %% [Block 79] 02-08-AS 실전 — 공공데이터로 지역별 전기차·충전기 분석하기 🔴
import numpy as np
import pandas as pd

regions = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종', '경기',
           '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']
base_2018 = [9000, 2500, 6000, 1800, 2000, 1500, 600, 400, 6500,
             1200, 1100, 1500, 1000, 2500, 1800, 2800, 13000]       # 2018년 1월 등록 대수
growth = [1.42, 1.60, 1.38, 1.65, 1.45, 1.50, 1.60, 1.55, 1.75,
          1.50, 1.55, 1.60, 1.50, 1.40, 1.55, 1.60, 1.25]           # 연간 증가 배율

def ev_table(months):
    """월말 날짜 목록을 받아 지역별 전기차 대수 표를 만든다 (최신 달이 맨 위)"""
    rows = []
    for m in months:
        t = (m.year - 2018) + (m.month - 1) / 12                    # 2018년 1월부터 지난 햇수
        wave = 1 + 0.06 * np.sin(2.3 * t)                            # 해마다 조금씩 다른 성장 속도
        counts = [int(b * g ** t * wave) for b, g in zip(base_2018, growth)]
        rows.append([m.strftime('%Y-%m-%d')] + counts + [sum(counts)])
    return pd.DataFrame(rows[::-1], columns=['기준일'] + regions + ['합계'])

old_months = pd.date_range('2018-01-31', '2022-06-30', freq='ME')     # 옛 파일: 2018-01 ~ 2022-06
new_months = pd.date_range('2022-04-30', '2023-03-31', freq='ME')     # 새 파일: 2022-04 ~ 2023-03 (3달 겹침)
ev_table(old_months).to_csv('지역별_전기차_현황.csv', index=False)                         # utf-8
ev_table(new_months).to_csv('지역별_전기차_현황_20230331.csv', index=False, encoding='cp949')

# 충전기 현황: 지역이 행, 연도가 열 (가나다순 지역 정렬)
st_base = [b * 0.12 for b in base_2018]                              # 2018년 충전기 수
st_growth = [1.95, 1.70, 1.80, 1.75, 1.85, 1.90, 1.65, 2.00, 1.80,
             1.95, 1.90, 1.85, 1.95, 1.80, 1.75, 1.70, 1.60]        # 연간 증가 배율
st = pd.DataFrame({'지역': regions})
for y in range(2015, 2024):
    t = (y - 2018) if y < 2023 else 4.55                            # 2023년은 7월 기준 → 반년 남짓만 증가
    wave = 1 + 0.08 * np.sin(1.7 * t)
    st[str(y)] = [int(sb * g ** t * wave) + 10 for sb, g in zip(st_base, st_growth)]
st.sort_values('지역').to_csv('지역별_충전기_현황_20230718.csv', index=False)
print("실습 파일 3개 생성 완료")


# %% [Block 80] 02-08-AS 실전 — 공공데이터로 지역별 전기차·충전기 분석하기 🔴
import pandas as pd

df1 = pd.read_csv('지역별_전기차_현황.csv')                                # utf-8 (기본)
df2 = pd.read_csv('지역별_전기차_현황_20230331.csv', encoding='cp949')     # 윈도우 한글
station = pd.read_csv('지역별_충전기_현황_20230718.csv')

print("df1:", df1.shape, "/ df2:", df2.shape, "/ station:", station.shape)
print(df1[['기준일', '서울', '경기', '제주', '합계']].head(4))
print("-" * 50)
print(df2[['기준일', '서울', '경기', '제주', '합계']].tail(3))


# %% [Block 81] 02-08-AS 실전 — 공공데이터로 지역별 전기차·충전기 분석하기 🔴
df1 = df1.drop([0, 1, 2], axis=0)                                  # 새 파일과 겹치는 최신 3개월 제거
ev = pd.concat([df1, df2])                                         # 위아래로 이어 붙이기
ev = ev.sort_values(by='기준일', ascending=True, ignore_index=True)  # 오래된 달부터, 번호 새로

print("합친 행 수:", len(ev), "/ 중복된 기준일:", ev['기준일'].duplicated().sum())
print(ev[['기준일', '서울', '합계']].head(2))
print(ev[['기준일', '서울', '합계']].tail(2))


# %% [Block 82] 02-08-AS 실전 — 공공데이터로 지역별 전기차·충전기 분석하기 🔴
ev['기준일'] = pd.to_datetime(ev['기준일']).dt.to_period('M')    # '2018-01-31' → 2018-01
ev = ev.drop(columns='합계')                                     # 지역별 분석에 합계는 방해
ev = ev.rename(columns={'기준일': '연도'})                        # 충전기 표와 열 이름 통일
print(ev.iloc[:3, :6])
print(ev['연도'].dtype)


# %% [Block 83] 02-08-AS 실전 — 공공데이터로 지역별 전기차·충전기 분석하기 🔴
print(station.iloc[:3, :6])          # 지역이 행, 연도가 열 — 전기차 표와 방향이 반대
print("-" * 50)
station = station.set_index('지역').T     # 지역을 이름표로 올린 뒤 전치 → 연도가 행, 지역이 열
station.columns.name = None               # 열 이름표 묶음의 이름('지역') 지우기
station.index.name = '연도'               # 행 이름표 묶음에 '연도'라는 이름
station = station.reset_index()           # 연도를 일반 열로
station['연도'] = station['연도'].astype('int64')   # '2015'(글자) → 2015(정수)
print(station.iloc[:3, :6])


# %% [Block 84] 02-08-AS 실전 — 공공데이터로 지역별 전기차·충전기 분석하기 🔴
print("열 순서 같은가?", list(station.columns) == list(ev.columns))
station = station[ev.columns]                                      # 전기차 표의 열 순서대로 다시 배열
print("맞춘 뒤      :", list(station.columns) == list(ev.columns))

station = station[station['연도'] >= 2018].reset_index(drop=True)   # 전기차 자료가 없는 2015~2017 제외
print(station[['연도', '서울', '경기', '제주']])


# %% [Block 85] 02-08-AS 실전 — 공공데이터로 지역별 전기차·충전기 분석하기 🔴
ev_final = ev.groupby(ev['연도'].dt.year).tail(1)      # 해마다 마지막 달의 행만 (누적 등록 대수이므로)
ev_final = ev_final.reset_index(drop=True)
print(ev_final[['연도', '서울', '경기', '제주']])
print("-" * 50)
ev_final['연도'] = ev_final['연도'].dt.year              # 2018-12 → 2018
print(ev_final[['연도', '서울', '경기', '제주']].dtypes)


# %% [Block 86] 02-08-AS 실전 — 공공데이터로 지역별 전기차·충전기 분석하기 🔴
recent_year = ev_final['연도'].max()
recent_ev = ev_final[ev_final['연도'] == recent_year].drop(columns='연도').iloc[0]
recent_station = station[station['연도'] == recent_year].drop(columns='연도').iloc[0]

ratio = (recent_ev / recent_station).round(2).sort_values()     # 같은 지역 이름표끼리 나누기
ratio.name = '충전기 1기당 전기차'
print(f"{recent_year}년 충전기 1기당 전기차 대수 (낮을수록 여유)")
print(ratio)


# %% [Block 87] 02-08-AS 실전 — 공공데이터로 지역별 전기차·충전기 분석하기 🔴
import plotly.express as px

fig = px.bar(x=ratio.index, y=ratio.values, color_discrete_sequence=['teal'],
             labels={'x': '지역', 'y': '충전기 1기당 전기차(대)'},
             title=f'{recent_year}년 지역별 전기차 수 대비 충전기 비율')
fig.update_xaxes(tickangle=-45)        # plt.xticks(rotation=45) 역할
fig.update_yaxes(showgrid=True)
fig.show()


# %% [Block 88] 02-08-AS 실전 — 공공데이터로 지역별 전기차·충전기 분석하기 🔴
seoul_ev = ev_final.set_index('연도')['서울']
seoul_station = station.set_index('연도')['서울']

seoul_ev_growth = (seoul_ev.pct_change() * 100).round(1)          # 전년 대비 증가율(%)
seoul_station_growth = (seoul_station.pct_change() * 100).round(1)
print(pd.DataFrame({'전기차': seoul_ev, '충전기': seoul_station,
                    '전기차 증가율(%)': seoul_ev_growth, '충전기 증가율(%)': seoul_station_growth}))


# %% [Block 89] 02-08-AS 실전 — 공공데이터로 지역별 전기차·충전기 분석하기 🔴
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=seoul_ev_growth.index, y=seoul_ev_growth, name='전기차 성장률',
                         mode='lines+markers', line=dict(dash='solid')))
fig.add_trace(go.Scatter(x=seoul_station_growth.index, y=seoul_station_growth, name='충전기 확장 속도',
                         mode='lines+markers', line=dict(dash='dash')))
fig.update_layout(title='서울 전기차 성장률 vs 충전기 확장 속도', xaxis_title='연도', yaxis_title='증가율(%)')
fig.update_xaxes(tickmode='array', tickvals=list(seoul_ev_growth.index))   # 연도를 정수 눈금으로
fig.show()
