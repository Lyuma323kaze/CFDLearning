import numpy as np
import matplotlib.pyplot as plt
import os

folder = 'Proj1'

a = 1
b = 1

# 定义两个函数（请替换为您的函数）
def Rek(k):
    return 2 * (0.5 * a * np.sin(3*k) + (- 2 * a - 1 / 12) * np.sin(2*k) + (2.5 * a + 2/3) * np.sin(k)) # 示例函数1：正弦函数

def Imk(k):
    return -2 * (-0.5 * b * np.cos(3*k) + 3 * b * np.cos(2*k) - 7.5 * b * np.cos(k))  # 示例函数2：余弦函数

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
plt.title(fr"Re(k) & Im(k) on (0, π) @ $\alpha = {a:.3f}, \beta = {b:.3f}$", fontsize=14)
plt.xlabel("k", fontsize=12)
plt.ylabel("f(k)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, np.pi)

# 设置π刻度
plt.xticks(
    ticks=[0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
    labels=['0', 'π/4', 'π/2', '3π/4', 'π']
)

# 添加图例
plt.legend(loc='upper right', fontsize=12)

file_path = os.path.join(folder, f'Re&Im@a={a:.3f}@b={b:.3f}.png')
plt.savefig(file_path)
plt.show()