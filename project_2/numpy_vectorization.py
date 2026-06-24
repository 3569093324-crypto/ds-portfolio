#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 15: NumPy 向量化运算
核心理念：用数组操作替代 for 循环 — 速度提升 10-500x
"""

import numpy as np
import time

# ============================================================
# 1. 创建 ndarray
# ============================================================
print("=" * 60)
print("  1. 创建 ndarray")
print("=" * 60)

# 全0 / 全1
zeros_arr = np.zeros((3, 4))
ones_arr  = np.ones((2, 5))
print(f"zeros(3,4):\n{zeros_arr}")
print(f"ones(2,5):\n{ones_arr}")

# 等差序列
arange_arr = np.arange(0, 20, 3)       # start, stop, step
linspace_arr = np.linspace(0, 1, 6)    # start, stop, num (含端点)
print(f"arange(0,20,3):  {arange_arr}")
print(f"linspace(0,1,6): {linspace_arr}")

# 随机数组
np.random.seed(42)
rand_uniform = np.random.rand(3, 3)           # [0,1) 均匀分布
rand_normal  = np.random.randn(3, 3)           # 标准正态分布
rand_int     = np.random.randint(1, 100, (2, 4))  # [1,100) 整数
print(f"rand(3,3):\n{rand_uniform}")
print(f"randint(1,100,(2,4)):\n{rand_int}")

# 单位矩阵 / 对角矩阵
eye_arr = np.eye(3)
diag_arr = np.diag([1, 5, 9])
print(f"eye(3):\n{eye_arr}")


# ============================================================
# 2. 布尔索引和花式索引
# ============================================================
print("\n" + "=" * 60)
print("  2. 布尔索引 & 花式索引 (No for loop!)")
print("=" * 60)

# 创建 100 个随机分数的数组
scores = np.random.randint(30, 100, 20)
print(f"原始分数: {scores}")

# 布尔索引：找出所有 >= 60 的分数（及格）
passing = scores[scores >= 60]
print(f"及格分数 (>=60): {passing}")

# 花式索引：取特定位置的元素
indices = [0, 3, 7, 10, 15]
selected = scores[indices]
print(f"选定位置{indices}: {selected}")

# 组合布尔条件：60-80 分
mid_range = scores[(scores >= 60) & (scores < 80)]
print(f"中等分数 [60,80): {mid_range}")

# 花式索引 + 赋值
arr = np.arange(10) ** 2
print(f"原数组: {arr}")
arr[[1, 3, 5, 7]] = -999
print(f"修改后: {arr}")


# ============================================================
# 3. Broadcasting 规则
# ============================================================
print("\n" + "=" * 60)
print("  3. Broadcasting 广播规则")
print("=" * 60)

# Broadcasting 规则（从后往前对齐维度）:
# 1. 维度相同 → 直接运算
# 2. 一个维度为1 → 拉伸匹配
# 3. 维度缺失 → 在前面补1

print("\n  规则示例:")
a = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
b = np.array([10, 20, 30])             # shape (3,) → 视为 (1, 3)
print(f"  a (2,3):\n{a}")
print(f"  b (3,): {b}")
print(f"  a + b (b广播为2行):\n{a + b}")
# b 被广播为 [[10,20,30],[10,20,30]]

c = np.array([[100], [200]])           # shape (2, 1)
print(f"\n  c (2,1):\n{c}")
print(f"  a + c (c广播为3列):\n{a + c}")
# c 被广播为 [[100,100,100],[200,200,200]]

print(f"\n  a + b + c:\n{a + b + c}")
# a(2,3) + b(3) → b广播到(2,3) → + c(2,1) → c广播到(2,3)


# ============================================================
# 4. 向量化替代 for 循环：欧氏距离矩阵
# ============================================================
print("\n" + "=" * 60)
print("  4. 欧氏距离矩阵 — for vs 向量化")
print("=" * 60)

# 生成 500 个随机二维点
N = 500
np.random.seed(42)
points = np.random.randn(N, 2)

# --- 方法1: for 循环 ---
def pairwise_distances_loop(X):
    n = len(X)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.sqrt(np.sum((X[i] - X[j]) ** 2))
    return D

# --- 方法2: 向量化 (利用 broadcasting) ---
def pairwise_distances_vec(X):
    # X: (N, D)
    # X[:, None, :] : (N, 1, D)
    # X[None, :, :] : (1, N, D)
    # diff: (N, N, D) via broadcasting
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))

start = time.perf_counter()
D_loop = pairwise_distances_loop(points[:50])  # 只用50个点（否则for循环太慢）
t_loop = time.perf_counter() - start
print(f"  for 循环 (50点):  {t_loop*1000:.1f} ms")

start = time.perf_counter()
D_vec = pairwise_distances_vec(points)         # 全量500点
t_vec = time.perf_counter() - start
print(f"  向量化   (500点):  {t_vec*1000:.1f} ms")
print(f"  速度对比: 向量化处理10x数据量，耗时仅 {t_loop/t_vec:.0f}x for循环")

# 验证正确性
assert np.allclose(D_loop, D_vec[:50, :50], atol=1e-10)
print(f"  ✅ 结果验证通过")


# ============================================================
# 5. 聚合操作 (axis 参数)
# ============================================================
print("\n" + "=" * 60)
print("  5. 聚合操作 — sum, mean, std (axis)")
print("=" * 60)

data = np.random.randn(4, 5)
print(f"数据 (4,5):\n{np.round(data, 2)}")

print(f"\n  axis=0 (沿列方向，压缩行 → 每列统计):")
print(f"  sum:  {np.round(data.sum(axis=0), 2)}")
print(f"  mean: {np.round(data.mean(axis=0), 2)}")
print(f"  std:  {np.round(data.std(axis=0), 2)}")

print(f"\n  axis=1 (沿行方向，压缩列 → 每行统计):")
print(f"  sum:  {np.round(data.sum(axis=1), 2)}")
print(f"  mean: {np.round(data.mean(axis=1), 2)}")

print(f"\n  axis=None (全数组统计):")
print(f"  sum:  {data.sum():.3f}")
print(f"  mean: {data.mean():.3f}")
print(f"  std:  {data.std():.3f}")
print(f"  min:  {data.min():.3f}")
print(f"  max:  {data.max():.3f}")
print(f"  argmax: {data.argmax()} (展平索引)")


# ============================================================
# 6. np.where() 条件赋值
# ============================================================
print("\n" + "=" * 60)
print("  6. np.where() — 向量化的 if-else")
print("=" * 60)

# 场景：给100个学生的成绩打分 A/B/C/D/F
grades = np.random.randint(0, 101, 15)
print(f"原始分数: {grades}")

# np.where(condition, 真值, 假值) — 可以嵌套！
letter_grades = np.where(
    grades >= 90, 'A',
    np.where(grades >= 80, 'B',
    np.where(grades >= 70, 'C',
    np.where(grades >= 60, 'D', 'F')))
)
for g, l in zip(grades, letter_grades):
    print(f"  {g:3d} → {l}")

# 对比 Python if-else（需要 for 循环）:
print(f"\n  Python if-else 写法（需要循环）:")
print(f"  for g in grades: 'A' if g>=90 else 'B' if g>=80 ...")


# ============================================================
# 7. 实战：Z-score 标准化 — for循环 vs 向量化 速度对比
# ============================================================
print("\n" + "=" * 60)
print("  7. Z-score 标准化 — 10,000行 × 20列")
print("=" * 60)

# 生成数据：10000行 × 20列
N_ROWS, N_COLS = 10000, 20
np.random.seed(42)
big_data = np.random.randn(N_ROWS, N_COLS) * 50 + 100  # 均值100, 标准差50

print(f"  数据形状: {big_data.shape}")

# --- 方法1: for 循环逐列标准化 ---
def zscore_loop(data):
    result = np.zeros_like(data)
    for col in range(data.shape[1]):
        col_mean = np.mean(data[:, col])
        col_std  = np.std(data[:, col])
        for row in range(data.shape[0]):
            result[row, col] = (data[row, col] - col_mean) / col_std
    return result

# --- 方法2: 向量化 (broadcasting) ---
def zscore_vec(data):
    mean = data.mean(axis=0)      # shape (20,)
    std  = data.std(axis=0)       # shape (20,)
    return (data - mean) / std    # broadcasting: (10000,20) 与 (20,) 运算

# 计时对比
start = time.perf_counter()
z1 = zscore_vec(big_data)
t_vec_z = time.perf_counter() - start

start = time.perf_counter()
z2 = zscore_loop(big_data)
t_loop_z = time.perf_counter() - start

print(f"\n  for 循环 (逐列+逐行): {t_loop_z*1000:.1f} ms")
print(f"  向量化 (broadcasting): {t_vec_z*1000:.1f} ms")
print(f"  🚀 向量化提速: {t_loop_z/t_vec_z:.0f}x")

# 验证
assert np.allclose(z1, z2, atol=1e-12)
print(f"  ✅ 结果验证通过")

# 验证标准化结果：每列均值 ≈ 0，标准差 ≈ 1
print(f"  标准化后 列均值范围: [{z1.mean(axis=0).min():.2e}, {z1.mean(axis=0).max():.2e}]")
print(f"  标准化后 列标准差范围: [{z1.std(axis=0).min():.4f}, {z1.std(axis=0).max():.4f}]")


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("  Day 15 总结：NumPy 向量化核心概念")
print("=" * 60)
print("""
  1. Broadcasting: 不同形状的数组自动对齐运算
  2. 布尔索引: arr[arr > threshold] — 无需 for 循环
  3. 向量化: 在 C 层面执行循环，比 Python for 快 10-500x
  4. axis 参数: axis=0 沿列方向（压缩行），axis=1 沿行方向
  5. np.where: 向量化的 if-else，可嵌套
  6. 内存布局: NumPy 按 C 顺序存储，axis=1 访问比 axis=0 快

  面试金句：
  "NumPy 的向量化操作将循环下沉到 C/Fortran 层执行，
   避免了 Python 解释器的循环开销，对于 10000 行数据，
   典型加速比在 50-500 倍之间。"
""")
