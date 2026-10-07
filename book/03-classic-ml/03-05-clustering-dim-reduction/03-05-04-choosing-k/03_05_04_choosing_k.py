# -*- coding: utf-8 -*-
"""
03-05-04 클러스터 개수 $K$ 결정하기 — (Choosing the Number of Clusters - Elbow Method) - 팔꿈치가 꺾이는 지점을 찾는 눈

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-05장 - 클러스터링과 차원축소/03-05-04 클러스터 개수 K 결정하기.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] 코드
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=150, n_features=2, centers=3,
                  cluster_std=0.5, shuffle=True, random_state=0)

distortions = []                    # K별 왜곡 값을 담을 빈 리스트

# range(1, 11) 은 1,2,...,10 입니다. 끝값 11은 포함되지 않습니다(파이썬 규칙).
for i in range(1, 11):
    km = KMeans(
        n_clusters=i,               # ★ 반복문 변수를 그대로 K로 사용
        init='random',            # 강의와 동일한 무작위 초기화
        n_init=10,                 # ★ 10번 재시작 — 곡선이 매끈해지는 비결!
        max_iter=300,               # 한 번 실행당 최대 반복
        tol=1e-04,                  # 1e-04 = 0.0001 (지수 표기법)
        random_state=0              # 재현성 고정
    )
    km.fit(X)                       # 학습 실행 (X만 넣습니다. y는 없습니다!)

    # km.inertia_ : 군집 내 제곱합(WCSS). 언더스코어(_)로 끝나는 속성은
    #               "학습 후에 만들어진 결과값"이라는 sklearn의 약속입니다.
    #               (예: labels_, cluster_centers_, inertia_)
    distortions.append(km.inertia_) # 군집 내 분산, 적을수록 좋음

# ── 급격하게 줄어드는 부분을 눈으로 찾기 ─────────────────────
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

# 숫자로도 확인 — "얼마나 줄었나"를 함께 보면 훨씬 명확합니다
for i, d in enumerate(distortions, start=1):
    gain = "" if i == 1 else f"(직전 대비 -{distortions[i-2]-d:.2f})"
    print(f"K={i:2d}  distortion={d:8.2f}  {gain}")


# %% [Block 2] 코드와 실행 결과
from sklearn.metrics import silhouette_score

# 실루엣은 "군집 간 비교"가 필요하므로 K=1 에서는 계산할 수 없습니다.
# (옆 반이 없으면 b 를 구할 수 없으니까요) → K=2 부터 시작
for k in range(2, 11):
    km = KMeans(n_clusters=k, init='random', n_init=10,
                max_iter=300, tol=1e-04, random_state=0).fit(X)

    # silhouette_score 는 모든 점의 s 를 구해 평균낸 값을 돌려줍니다.
    # 인자로 (원본 데이터, 군집 라벨) 두 개가 필요합니다.
    score = silhouette_score(X, km.labels_)
    print(f"K={k:2d}  distortion={km.inertia_:8.2f}  silhouette={score:.4f}")
