# -*- coding: utf-8 -*-
"""
depth 후보를 하나씩 바꿔가며 학습/검증 정확도를 비교합니다

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-99 부록 전통 머신러닝으로 텍스트 분류하기 — 딥러닝 이전의 무기들/03-99-03 ⚖ 과소적합·과대적합과 하이퍼파라미터 튜닝 — 옷 맞춤 비유.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 3단계 — 코드로 직접 확인하기
from sklearn.tree import DecisionTreeClassifier

# depth 후보를 하나씩 바꿔가며 학습/검증 정확도를 비교합니다
for depth in [1, 3, 5, 10, None]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(Xtr, y_train)              # 학습 데이터로 학습
    train_acc = model.score(Xtr, y_train)  # 학습 데이터 자체 성능 (암기 여부 확인)
    val_acc = model.score(Xval, y_val)     # 처음 보는 데이터 성능 (일반화 여부 확인)
    print(f"depth={depth}: train={train_acc:.3f}, val={val_acc:.3f}")

# depth=1:    train=0.65, val=0.64   (과소적합 — 둘 다 낮음)
# depth=5:    train=0.85, val=0.82   (적절!)
# depth=None: train=1.00, val=0.71   (과대적합 — 차이가 큼)


# %% [Block 2] 4단계 — 하이퍼파라미터 튜닝: GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# 탐색할 하이퍼파라미터 후보들 (옷의 치수 후보라고 생각하세요)
param_grid = {
    'n_estimators': [50, 100, 200],   # 트리 개수
    'max_depth': [5, 10, None],       # 트리 깊이
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,                # 5-fold 교차검증
    scoring='f1',
    n_jobs=-1,           # 가능한 CPU 코어를 모두 사용
)
grid.fit(Xtr, y_train)

print("최적 하이퍼파라미터:", grid.best_params_)
print("최적 CV 점수:", grid.best_score_)
# 예: {'n_estimators': 100, 'max_depth': 10}, 0.87
