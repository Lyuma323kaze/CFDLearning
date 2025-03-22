import numpy as np
import matplotlib.pyplot as plt

# 定义两个函数（请替换为您的函数）
def Rek(k):
    return (8 * np.sin(k) - np.sin(2 * k)) / 6  # 示例函数1：正弦函数

def Imk(k):
    return -(3 - 4 * np.cos(k) + np.cos(2 * k)) / 6  # 示例函数2：余弦函数

# 生成数据点
x = np.linspace(0, np.pi, 300)
y1 = Rek(x)
y2 = Imk(x)

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制两条曲线
plt.plot(x, y1, color='blue', linewidth=2, label='Re(k)')   # 第一条曲线
plt.plot(x, y2, color='red',  linewidth=2, linestyle='--', label='Im(k)')  # 第二条曲线

# 设置图形属性
plt.title("Function Comparison on (0, π)", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("f(x)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, np.pi)

# 设置π刻度
plt.xticks(
    ticks=[0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
    labels=['0', 'π/4', 'π/2', '3π/4', 'π']
)

# 添加图例
plt.legend(loc='upper right', fontsize=12)

# 显示图形
plt.show()