# -*- coding: utf-8 -*-
"""
1단계: 데이터 생성

출처: 「Zero to 머신러닝 딥러닝 Master」 (WikiDocs https://wikidocs.net/book/21464)
원고: 03권  초기(전통적) 머신러닝으로 기초 다지기/03-05장 - 클러스터링과 차원축소.md

책에 실린 코드 블록을 순서대로 모은 스크립트입니다. (# %% 셀 구분은 VS Code / Jupytext 에서 셀로 인식됩니다)
"""

# %% [Block 1] ③ 다중 실행(Multiple Runs) — scikit-learn은 기본적으로 K-Means++를 사용합니다
# scikit-learn은 기본적으로 K-Means++를 사용합니다
from sklearn.cluster import KMeans

# n_init=10 : 다른 초기화로 10번 실행 후 최선 선택
# init='k-means++' : K-Means++ 초기화 (기본값)
kmeans = KMeans(
    n_clusters=3,           # 클러스터 개수 K=3
    init='k-means++',       # 스마트 초기화 방법
    n_init=10,              # 10번 다른 시작점으로 실행
    max_iter=300,           # 최대 반복 횟수
    random_state=42         # 재현성을 위한 난수 고정
)


# %% [Block 2] ① NumPy로 K-평균 직접 구현
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ─────────────────────────────────────────────
# 1단계: 데이터 생성
#   make_blobs와 유사하게 3개의 군집을 가진
#   가상 데이터를 직접 만듭니다.
# ─────────────────────────────────────────────
np.random.seed(42)  # 항상 같은 결과가 나오도록 난수 고정

# 군집 중심 3개를 설정합니다
centers = np.array([[2.0, 2.0],   # 군집A 중심
                    [7.0, 2.0],   # 군집B 중심
                    [4.5, 7.0]])  # 군집C 중심

# 각 중심 주위에 50개씩 총 150개 점 생성
X = np.vstack([
    centers[k] + np.random.randn(50, 2)  # 가우시안 노이즈 추가
    for k in range(3)
])
# X.shape = (150, 2) → 150개 점, 각 점은 2차원(x1, x2)

# ─────────────────────────────────────────────
# 2단계: K-평균 알고리즘 구현
# ─────────────────────────────────────────────
K = 3          # 클러스터 개수
max_iter = 100 # 최대 반복 횟수 (수렴 안 되면 여기서 멈춤)
tol = 1e-4     # 수렴 판정 기준 (센트로이드 이동이 이 값보다 작으면 종료)

# ── 초기화: K개의 데이터 포인트를 무작위로 센트로이드로 선택 ──
idx = np.random.choice(len(X), K, replace=False)
# replace=False: 같은 점이 중복 선택되지 않도록
centroids = X[idx].copy()
# centroids.shape = (3, 2) → 3개의 센트로이드, 각 2차원

labels = np.zeros(len(X), dtype=int)  # 각 점의 클러스터 번호 저장 공간

for iteration in range(max_iter):

    # ── 할당(Assignment) 단계 ──────────────────────
    # 각 데이터 포인트를 가장 가까운 센트로이드에 배정합니다
    for i, x in enumerate(X):
        # 각 센트로이드까지 유클리드 거리² 계산
        dists = np.sum((centroids - x) ** 2, axis=1)
        # (centroids - x): 센트로이드 - 현재 점 = 차이 벡터
        # ** 2: 제곱    sum(axis=1): 차원별 합산 → 거리²
        labels[i] = np.argmin(dists)
        # argmin: 가장 작은 거리의 인덱스 = 가장 가까운 클러스터 번호

    # ── 갱신(Update) 단계 ──────────────────────────
    # 각 클러스터의 새 센트로이드 = 클러스터 내 데이터들의 평균
    new_centroids = np.zeros_like(centroids)
    for k in range(K):
        cluster_points = X[labels == k]
        # labels == k: k번 클러스터에 속한 점들만 선택
        if len(cluster_points) > 0:
            new_centroids[k] = cluster_points.mean(axis=0)
            # axis=0: 행(데이터) 방향으로 평균 → 각 차원의 평균

    # ── 수렴 판단 ───────────────────────────────────
    # 센트로이드가 거의 움직이지 않으면 종료합니다
    shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
    # linalg.norm: 각 센트로이드의 이동 거리 계산
    # np.max: 가장 많이 이동한 센트로이드의 거리
    centroids = new_centroids.copy()
    if shift < tol:
        print(f"반복 {iteration+1}회에서 수렴 완료!")
        break

# ─────────────────────────────────────────────
# 3단계: 왜곡 비용(Inertia) 계산
# ─────────────────────────────────────────────
inertia = sum(
    np.sum((X[labels == k] - centroids[k]) ** 2)
    for k in range(K)
)
print(f"최종 왜곡 비용(Inertia): {inertia:.2f}")

# ─────────────────────────────────────────────
# 4단계: 시각화
# ─────────────────────────────────────────────
colors = ['#ef4444', '#3b82f6', '#22c55e']
plt.figure(figsize=(8, 6))
for k in range(K):
    pts = X[labels == k]
    plt.scatter(pts[:, 0], pts[:, 1],
                c=colors[k], alpha=0.6, s=50,
                label=f'클러스터 {k+1}')
    plt.scatter(centroids[k, 0], centroids[k, 1],
                c=colors[k], marker='X', s=200,
                edgecolor='black', linewidth=1.5, zorder=5)
plt.title('K-평균 클러스터링 결과 (NumPy 직접 구현)', fontsize=13)
plt.xlabel('특성 1'); plt.ylabel('특성 2')
plt.legend(); plt.tight_layout(); plt.show()


# %% [Block 3] ② scikit-learn으로 붓꽃 데이터 클러스터링
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
# 1단계: 데이터 준비
#   붓꽃(Iris) 데이터셋은 3종의 붓꽃 150개 샘플
#   꽃받침 길이/너비, 꽃잎 길이/너비 4개 특성
# ─────────────────────────────────────────────
iris = load_iris()
X = iris.data         # (150, 4) — 4개 특성
y_true = iris.target  # 실제 정답 (시각화 비교용)

# 특성 스케일 표준화: 단위가 다른 특성들을 동일한 스케일로 맞춤
# K-평균은 거리 기반이므로 스케일 맞추기가 매우 중요합니다!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# fit_transform: 평균을 0, 표준편차를 1로 변환

# ─────────────────────────────────────────────
# 2단계: 엘보우 방법으로 최적 K 탐색
# ─────────────────────────────────────────────
inertias = []        # 각 K에서의 왜곡 비용 저장
silhouettes = []     # 각 K에서의 실루엣 점수 저장
k_range = range(2, 9)  # K=2부터 8까지 시도

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++',
                n_init=10, random_state=42)
    km.fit(X_scaled)                        # K-평균 학습 실행
    inertias.append(km.inertia_)           # inertia_: 최종 왜곡 비용 J
    score = silhouette_score(X_scaled, km.labels_)
    silhouettes.append(score)
    print(f"K={k}: Inertia={km.inertia_:.1f}, Silhouette={score:.3f}")

# ─────────────────────────────────────────────
# 3단계: 최적 K=3으로 최종 클러스터링
# ─────────────────────────────────────────────
best_k = 3  # 엘보우·실루엣 분석 결과 K=3이 최적
km_final = KMeans(n_clusters=best_k, init='k-means++',
                  n_init=10, random_state=42)
km_final.fit(X_scaled)
labels = km_final.labels_       # 각 샘플의 클러스터 번호
centers = km_final.cluster_centers_  # 최종 센트로이드 좌표 (3, 4)

# ─────────────────────────────────────────────
# 4단계: 2D 시각화 (첫 두 주성분으로)
#   4차원 데이터를 그대로 그리기 어려우므로
#   꽃받침 길이(특성0)·꽃잎 길이(특성2)만 사용
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ['#ef4444', '#3b82f6', '#22c55e']

# 그래프 1: 엘보우 곡선
axes[0].plot(k_range, inertias, 'bo-', linewidth=2)
axes[0].axvline(x=3, color='red', linestyle='--', label='최적 K=3')
axes[0].set_title('엘보우 방법'); axes[0].set_xlabel('K')
axes[0].set_ylabel('왜곡 비용 (Inertia)'); axes[0].legend()

# 그래프 2: K-평균 결과
for k in range(best_k):
    mask = labels == k
    axes[1].scatter(X_scaled[mask, 0], X_scaled[mask, 2],
                   c=colors[k], alpha=0.7, s=50, label=f'클러스터{k+1}')
axes[1].scatter(centers[:, 0], centers[:, 2], c='black',
               marker='X', s=200, zorder=5, label='센트로이드')
axes[1].set_title('K-평균 클러스터링 (K=3)')
axes[1].set_xlabel('꽃받침 길이'); axes[1].set_ylabel('꽃잎 길이')
axes[1].legend()

# 그래프 3: 실제 정답과 비교
species = ['Setosa', 'Versicolor', 'Virginica']
for k in range(best_k):
    mask = y_true == k
    axes[2].scatter(X_scaled[mask, 0], X_scaled[mask, 2],
                   c=colors[k], alpha=0.7, s=50, label=species[k])
axes[2].set_title('실제 품종 (정답)')
axes[2].set_xlabel('꽃받침 길이'); axes[2].set_ylabel('꽃잎 길이')
axes[2].legend()
plt.tight_layout(); plt.show()


# %% [Block 4] ③ NumPy로 PCA 직접 구현
import numpy as np

class PCA_from_scratch:
    """
    PCA(주성분분석) 직접 구현 클래스
    
    학습 순서:
    1. 표준화 (fit)
    2. 공분산 행렬 계산
    3. 고유값·고유벡터 분해
    4. 상위 k개 주성분 선택
    5. 투영 (transform) / 복원 (inverse_transform)
    """
    
    def __init__(self, n_components):
        self.n_components = n_components   # 축소할 목표 차원 수
        self.components_ = None            # 주성분 벡터들 (저장 공간)
        self.explained_variance_ = None    # 각 주성분의 분산(고유값)
        self.explained_variance_ratio_ = None  # 설명 분산 비율
        self.mean_ = None                  # 학습 데이터의 평균 (복원시 필요)
    
    def fit(self, X):
        """
        PCA 학습: 주성분 방향을 찾습니다
        X: (n_samples, n_features) 형태의 데이터
        """
        n, d = X.shape
        # n = 샘플 수, d = 특성(차원) 수
        
        # ── 1단계: 평균 중심화 (Mean Centering) ──────────
        # 각 특성의 평균을 빼서 데이터 중심을 원점으로 이동
        # 이 단계가 없으면 공분산이 아닌 상관행렬이 됩니다
        self.mean_ = X.mean(axis=0)
        # axis=0: 행(샘플) 방향으로 평균 → 각 특성의 평균값
        X_centered = X - self.mean_
        # 브로드캐스팅: (n, d) - (d,) → (n, d)
        
        # ── 2단계: 공분산 행렬 계산 ─────────────────────
        # Σ = XᵀX / (n-1)
        # (d, n) @ (n, d) = (d, d) 행렬
        # n-1로 나누는 이유: 불편 추정량 (Bessel 보정)
        cov_matrix = (X_centered.T @ X_centered) / (n - 1)
        # cov_matrix.shape = (d, d)
        
        # ── 3단계: 고유값 분해 ───────────────────────────
        # np.linalg.eigh: 대칭 행렬의 고유값·고유벡터 계산
        # eigh는 eig보다 수치적으로 안정적 (대칭 행렬 전용)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # eigenvalues:  (d,)    — 고유값 (오름차순 정렬됨)
        # eigenvectors: (d, d)  — 고유벡터 (열벡터)
        
        # ── 4단계: 내림차순 정렬 ─────────────────────────
        # eigh는 오름차순으로 반환 → 큰 값부터 정렬해야 함
        # 분산(고유값)이 가장 큰 방향이 제1주성분!
        idx = np.argsort(eigenvalues)[::-1]
        # argsort: 정렬 인덱스 반환  [::-1]: 뒤집기 (내림차순)
        eigenvalues  = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # eigenvectors[:, idx]: 열 순서를 고유값 크기 순으로 재배열
        
        # ── 5단계: 상위 k개 주성분 선택 ─────────────────
        self.components_ = eigenvectors[:, :self.n_components].T
        # eigenvectors의 앞 k개 열 선택 후 전치
        # shape: (d, k).T = (k, d) → 각 행이 하나의 주성분 벡터
        
        self.explained_variance_ = eigenvalues[:self.n_components]
        # 상위 k개 고유값 = 각 주성분이 설명하는 분산
        
        self.explained_variance_ratio_ = (
            self.explained_variance_ / eigenvalues.sum()
        )
        # 전체 분산 대비 각 주성분의 설명 비율
        
        return self
    
    def transform(self, X):
        """
        새 데이터를 주성분 공간으로 투영 (차원 축소)
        반환: (n_samples, n_components) 형태의 저차원 데이터
        """
        X_centered = X - self.mean_
        # 학습 때의 평균으로 중심화 (일관성 유지!)
        return X_centered @ self.components_.T
        # (n, d) @ (d, k) = (n, k) → 저차원 좌표
    
    def inverse_transform(self, Z):
        """
        저차원 데이터를 원래 차원으로 복원 (근사값)
        정보 일부가 손실되므로 완전 복원은 불가능합니다
        """
        return Z @ self.components_ + self.mean_
        # (n, k) @ (k, d) + (d,) = (n, d) → 복원된 데이터
        # + self.mean_: 제거했던 평균을 다시 더해줌

# ─── 테스트 ───────────────────────────────────────
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data.astype(float)  # (150, 4)

pca = PCA_from_scratch(n_components=2)
pca.fit(X)

Z = pca.transform(X)         # (150, 4) → (150, 2)
X_restored = pca.inverse_transform(Z)  # (150, 2) → (150, 4)

print("원본 shape:", X.shape)
print("축소 shape:", Z.shape)
print("복원 shape:", X_restored.shape)
print("\n각 주성분 설명 분산 비율:", pca.explained_variance_ratio_)
print("누적 설명 분산 비율:", pca.explained_variance_ratio_.sum())

# 복원 오차 계산: 원본과 복원본의 차이
reconstruction_error = np.mean((X - X_restored) ** 2)
print(f"\n복원 오차(MSE): {reconstruction_error:.4f}")


# %% [Block 5] ① scikit-learn PCA + 붓꽃 전체 실습
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# ─────────────────────────────────────────────
# 1단계: 데이터 로드 및 표준화
# ─────────────────────────────────────────────
iris = load_iris()
X = iris.data    # (150, 4) — 4개 특성
y = iris.target  # (150,)  — 3개 클래스 (0, 1, 2)
target_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

# 표준화는 PCA 전 필수! 단위가 다른 특성들이 공정하게 반영됨
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────
# 2단계: 전체 PCA (모든 주성분 계산)
#   먼저 전체 주성분으로 분산 기여도를 확인합니다
# ─────────────────────────────────────────────
pca_full = PCA()                  # n_components 지정 안 하면 전체 계산
pca_full.fit(X_scaled)

evr = pca_full.explained_variance_ratio_
# evr: 각 주성분의 설명 분산 비율 (내림차순)
cumulative_evr = np.cumsum(evr)
# cumsum: 누적 합계 → 몇 개 주성분으로 몇 %를 설명하는지

print("각 PC의 설명 분산 비율:")
for i, (r, cr) in enumerate(zip(evr, cumulative_evr)):
    print(f"  PC{i+1}: {r:.4f} ({r*100:.2f}%)  누적: {cr*100:.2f}%")

# ─────────────────────────────────────────────
# 3단계: 2D PCA — 시각화를 위해 2개 주성분으로 축소
# ─────────────────────────────────────────────
pca_2d = PCA(n_components=2)      # 2개 주성분만 사용
X_pca = pca_2d.fit_transform(X_scaled)
# fit_transform = fit(학습) + transform(변환)을 한 번에
# X_pca.shape = (150, 2) — 4D → 2D 압축 완료

print(f"\n2D PCA 설명 분산 비율: {pca_2d.explained_variance_ratio_}")
print(f"누적: {pca_2d.explained_variance_ratio_.sum():.4f}")

# ─────────────────────────────────────────────
# 4단계: 주성분 로딩(적재값) 분석
#   각 원래 특성이 주성분에 얼마나 기여하는지 확인
# ─────────────────────────────────────────────
loadings = pca_2d.components_
# loadings.shape = (2, 4) — 2개 PC × 4개 특성
feature_names = iris.feature_names

print("\n주성분 로딩 행렬 (어떤 특성이 중요한가?):")
print(f"{'특성':30s}  PC1     PC2")
for name, l1, l2 in zip(feature_names, loadings[0], loadings[1]):
    print(f"  {name:30s}: {l1:+.3f}   {l2:+.3f}")

# ─────────────────────────────────────────────
# 5단계: 복원 및 복원 오차 계산
# ─────────────────────────────────────────────
X_restored = pca_2d.inverse_transform(X_pca)
# 2D → 4D 복원 (근사값이므로 완전 복원 불가)
recon_error = np.mean((X_scaled - X_restored) ** 2)
print(f"\n복원 오차(MSE): {recon_error:.4f}")

# ─────────────────────────────────────────────
# 6단계: 시각화 4종
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
colors = ['#ef4444', '#3b82f6', '#22c55e']
markers = ['o', 's', '^']

# 그래프 1: Scree Plot (설명 분산 비율)
ax = axes[0, 0]
ax.bar(range(1, 5), evr * 100, color='#3b82f6', alpha=0.7, label='개별')
ax.plot(range(1, 5), cumulative_evr * 100, 'ro-', linewidth=2, label='누적')
ax.axhline(95, color='green', linestyle='--', label='95% 기준선')
ax.set_title('Scree Plot — 설명 분산 비율')
ax.set_xlabel('주성분 번호'); ax.set_ylabel('분산 설명 비율 (%)')
ax.legend(); ax.set_xticks(range(1, 5))

# 그래프 2: 2D PCA 산점도
ax = axes[0, 1]
for i, name in enumerate(target_names):
    mask = y == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colors[i], marker=markers[i], s=60,
               alpha=0.8, label=name)
ax.set_title(f'PCA 2D 투영\n(설명 분산: {pca_2d.explained_variance_ratio_.sum():.1%})')
ax.set_xlabel(f'PC1 ({evr[0]:.1%})')
ax.set_ylabel(f'PC2 ({evr[1]:.1%})')
ax.legend()

# 그래프 3: 주성분 로딩 히트맵
ax = axes[1, 0]
im = ax.imshow(loadings, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(4)); ax.set_xticklabels(['꽃받침\n길이','꽃받침\n너비','꽃잎\n길이','꽃잎\n너비'])
ax.set_yticks([0, 1]); ax.set_yticklabels(['PC1', 'PC2'])
for i in range(2):
    for j in range(4):
        ax.text(j, i, f'{loadings[i,j]:+.2f}', ha='center', va='center', fontsize=11)
plt.colorbar(im, ax=ax); ax.set_title('주성분 로딩 (특성 기여도)')

# 그래프 4: 원본 vs 복원 (첫 번째 특성 비교)
ax = axes[1, 1]
ax.scatter(range(150), X_scaled[:, 0], s=15, alpha=0.6, label='원본', color='#3b82f6')
ax.scatter(range(150), X_restored[:, 0], s=15, alpha=0.6, label='복원', color='#ef4444')
ax.set_title(f'원본 vs 복원 (꽃받침 길이)\nMSE={recon_error:.4f}')
ax.set_xlabel('샘플 인덱스'); ax.set_ylabel('표준화 값'); ax.legend()

plt.tight_layout(); plt.show()


# %% [Block 6] ② K-평균 + PCA 결합 파이프라인
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

# Pipeline: 여러 단계를 하나로 묶어 관리합니다
# 순서대로 실행: 표준화 → PCA 축소 → K-평균 클러스터링
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 1단계: 표준화
    ('pca',    PCA(n_components=2)), # 2단계: 4D → 2D 압축
    ('kmeans', KMeans(n_clusters=3,  # 3단계: 3개 클러스터
                      init='k-means++',
                      n_init=10,
                      random_state=42))
])

pipeline.fit(X)               # 전체 파이프라인 학습
labels = pipeline.predict(X)  # 클러스터 번호 예측

# PCA로 변환된 2D 데이터 가져오기 (시각화용)
X_pca = pipeline.named_steps['pca'].transform(
    pipeline.named_steps['scaler'].transform(X)
)

# 성능 평가: 클러스터 결과와 실제 정답 비교
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(y, labels)
# ARI: -1~1, 1=완벽 일치, 0=무작위, 레이블 순서 무관
print(f"ARI(조정 랜드 지수): {ari:.3f}")
# 1에 가까울수록 클러스터링이 실제 품종과 잘 일치
