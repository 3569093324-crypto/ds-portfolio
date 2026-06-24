#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 48: 聚类分析 — K-Means & DBSCAN
肘部法则 · 轮廓系数 · PCA可视化 · 用户分群画像
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs, make_moons
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# 生成用户行为数据 (模拟电商用户)
n = 500
X_blob, _ = make_blobs(n_samples=n, n_features=6, centers=4,
                        cluster_std=[0.8, 1.2, 0.6, 1.5], random_state=42)
# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_blob)

feature_names = ['recency', 'frequency', 'monetary', 'avg_order',
                 'session_len', 'discount_rate']

print("=" * 60)
print("  聚类分析: K-Means & DBSCAN")
print("=" * 60)
print(f"  数据: {n} users × 6 features")
print(f"  True clusters: 4 (known for validation)")

# ============================================================
# 1 & 2. K-Means + 肘部曲线 + 轮廓系数
# ============================================================
K_range = range(2, 11)
inertias = []
silhouettes = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

best_k_sil = K_range[np.argmax(silhouettes)]

print(f"\n  {'K':5s} | {'Inertia':10s} | {'Silhouette':12s}")
print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*12}")
for k, inert, sil in zip(K_range, inertias, silhouettes):
    marker = ' ← BEST' if k == best_k_sil else ''
    print(f"  {k:5d} | {inert:10.1f} | {sil:12.4f}{marker}")
print(f"\n  Elbow K ≈ 4 (from inertia curve)")
print(f"  Best Silhouette K = {best_k_sil}")

# ============================================================
# K-Means with best K
# ============================================================
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# ============================================================
# 4. 聚类画像
# ============================================================
df_clusters = pd.DataFrame(X_scaled, columns=feature_names)
df_clusters['cluster'] = cluster_labels

print(f"\n  Cluster Profiles (mean values):")
cluster_profiles = df_clusters.groupby('cluster').mean().round(3)
print(cluster_profiles.to_string())
print(f"\n  Cluster sizes: {df_clusters['cluster'].value_counts().sort_index().to_dict()}")

# ============================================================
# 5. DBSCAN
# ============================================================
# 同时用 make_moons 展示 DBSCAN 处理非球形簇的优势
X_moons, _ = make_moons(n_samples=300, noise=0.08, random_state=42)

dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise_db = list(dbscan_labels).count(-1)

dbscan_moons = DBSCAN(eps=0.2, min_samples=5)
dbscan_moons_labels = dbscan_moons.fit_predict(X_moons)

print(f"\n  DBSCAN: {n_clusters_db} clusters found, {n_noise_db} noise points")
print(f"  K-Means is forced to find 4 clusters; DBSCAN found {n_clusters_db} naturally")

# ============================================================
# 7. 聚类标签作为特征
# ============================================================
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

# 测试: 加聚类特征是否有帮助
X_clf, y_clf = make_classification(n_samples=500, n_features=8, random_state=42)
X_clf_scaled = StandardScaler().fit_transform(X_clf)
km_clf = KMeans(n_clusters=5, random_state=42, n_init=10)
cluster_feat = km_clf.fit_predict(X_clf_scaled).reshape(-1, 1)
X_with_cluster = np.hstack([X_clf_scaled, cluster_feat])

cv_original = cross_val_score(LogisticRegression(max_iter=2000), X_clf_scaled, y_clf, cv=5, scoring='roc_auc')
cv_with_cluster = cross_val_score(LogisticRegression(max_iter=2000), X_with_cluster, y_clf, cv=5, scoring='roc_auc')
print(f"\n  Adding cluster as feature: CV AUC {cv_original.mean():.4f} → {cv_with_cluster.mean():.4f} "
      f"({'✅ improved' if cv_with_cluster.mean() > cv_original.mean() else 'no change'})")

# ============================================================
# 可视化
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# (1) 肘部曲线
ax = axes[0, 0]
ax.plot(K_range, inertias, 'o-', color='#2196F3', linewidth=2, markersize=8)
ax.axvline(4, color='red', linestyle='--', alpha=0.5, label='Elbow ≈ 4')
ax.set_xlabel('K'); ax.set_ylabel('Inertia (WCSS)')
ax.set_title('Elbow Method', fontweight='bold')
ax.legend()

# (2) 轮廓系数
ax = axes[0, 1]
ax.plot(K_range, silhouettes, 'o-', color='#4CAF50', linewidth=2, markersize=8)
ax.axvline(best_k_sil, color='red', linestyle='--', alpha=0.5, label=f'Best K={best_k_sil}')
ax.set_xlabel('K'); ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Analysis', fontweight='bold')
ax.legend()

# (3) PCA可视化聚类
ax = axes[0, 2]
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
colors_cluster = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
for i in range(4):
    mask = cluster_labels == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors_cluster[i],
               label=f'Cluster {i} ({mask.sum()})', alpha=0.6, s=30, edgecolors='white')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('K-Means Clusters (PCA 2D)', fontweight='bold')
ax.legend(fontsize=7)

# (4) 聚类画像热力图
ax = axes[1, 0]
im = ax.imshow(cluster_profiles.values.T, cmap='RdBu_r', aspect='auto')
ax.set_xticks(range(4))
ax.set_xticklabels([f'Cluster {i}' for i in range(4)])
ax.set_yticks(range(6))
ax.set_yticklabels(feature_names, fontsize=9)
for i in range(4):
    for j in range(6):
        ax.text(i, j, f'{cluster_profiles.values[i, j]:.2f}',
                ha='center', va='center', fontsize=7, fontweight='bold',
                color='white' if abs(cluster_profiles.values[i, j]) > 0.5 else 'black')
ax.set_title('Cluster Profiles Heatmap\n(Standardized means)', fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)

# (5) DBSCAN vs K-Means on moons
ax = axes[1, 1]
# K-Means on moons
km_moons = KMeans(n_clusters=2, random_state=42, n_init=10)
km_m_labels = km_moons.fit_predict(X_moons)
ax.scatter(X_moons[:, 0], X_moons[:, 1], c=km_m_labels, cmap='Set1',
           alpha=0.6, s=20, edgecolors='white')
ax.set_title('K-Means on Moon Data\n(Fails — forces spherical clusters)',
             fontweight='bold', fontsize=9)

# (6) DBSCAN on moons
ax = axes[1, 2]
ax.scatter(X_moons[:, 0], X_moons[:, 1], c=dbscan_moons_labels, cmap='Set1',
           alpha=0.6, s=20, edgecolors='white')
n_noise_moons = list(dbscan_moons_labels).count(-1)
ax.set_title(f'DBSCAN on Moon Data\n(Success — finds non-spherical clusters)\n'
             f'{n_noise_moons} noise points', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'clustering_analysis.png'), dpi=150)
plt.close()
print("  Saved: clustering_analysis.png")

print(f"""
  K-Means vs DBSCAN:

  K-Means:
    ✅ Fast, scalable, easy to interpret
    ❌ Must specify K, assumes spherical clusters
    ✅ Use: Customer segmentation, document clustering

  DBSCAN:
    ✅ No need to specify K, finds arbitrary shapes
    ✅ Robust to outliers (marks as noise)
    ❌ Struggles with varying densities, high dimensions
    ✅ Use: Anomaly detection, geospatial clustering
""")

print("✅ Day 48 完成")
