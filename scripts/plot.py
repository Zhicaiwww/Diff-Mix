import matplotlib.pyplot as plt
font = {'family': 'serif',
        'font': 'Time New Roman',
        'weight': 'normal',
        'size': 16,
        }
# 示例数据
categories = ['2X', '4X', '6X', '8X', '10X']
values = [82.74, 82.93, 83.08, 83.15, 83.47]

# 创建带有正负值的条形图
plt.bar(categories, values, color=['#9FBB73' for v in values])

# 添加标签和标题
import matplotlib.pyplot as plt

# 示例数据
categories = ['2X', '4X', '6X', '8X', '10X']
values = [82.74, 82.93, 83.08, 83.15, 83.47]
threshold = 81.60

# 创建带有正负值的条形图
bars = plt.bar(categories, values, color=['#9FBB73'])

# 添加标签和标题
plt.xlabel('Synthetic data size',fontdict=font)
plt.ylabel('Acc (%)',fontdict=font)

# 在纵坐标上添加虚线
plt.axhline(y=threshold, linestyle='--', color='gray', label='Baseline')

# 设置纵坐标刻度范围
plt.ylim(81, 84)

# 设置纵坐标刻度
plt.yticks(range(81, 84))

# 在每个条形上方显示相对于81.60增加了多少
for bar in bars:
    height = bar.get_height()
    difference = height - threshold
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'+{difference:.2f}', ha='center', color='black',fontsize=12)

# 显示图例
plt.legend(fontsize=12)

# 显示图形
plt.show()




