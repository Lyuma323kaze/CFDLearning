import numpy as np

# 定义不等式中的两个绝对值表达式
def expr1(k):
    return np.abs(- 1 + (8 * np.sin(k) - np.sin(2 * k)) / (6 * k))

def expr2(k):
    return np.abs((3 - 4 * np.cos(k) + np.cos(2 * k)) / (6 * k))

# 生成密集采样点（避开k=0）
k = np.linspace(1e-8, 2 * np.pi, 100000)

# 计算表达式值
val1 = expr1(k)
val2 = expr2(k)
max_val = np.maximum(val1, val2)

# 结果判断
if not np.any(max_val <= 0.005):
    print("解集为空，无满足条件的k ∈ (0, π)")
else:
    valid_k = k[max_val <= 0.005]
    print(f"解集为: ({valid_k.min():.6f}, {valid_k.max():.6f})")

# 可视化验证（对数坐标系）
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.semilogy(k, val1, label=r'$|1+\frac{\sin k}{k}|$', alpha=0.8)
plt.semilogy(k, val2, label=r'$|\frac{1-\cos k}{k}|$', alpha=0.8)
plt.axhline(0.005, color='red', linestyle='--', label='阈值 0.005')
plt.title("表达式值变化趋势（对数坐标）")
plt.xlabel("k")
plt.ylabel("值")
plt.legend()
plt.show()