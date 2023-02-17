import matplotlib.pyplot as plt
import numpy as np
import yaml

with open('../../config.yml', 'r') as yml:
    config = yaml.safe_load(yml)
# データ生成
# x = np.linspace(0, 10, 100)
# y = x + np.random.randn(100) 

#ファイル読み込み
f = open(config["rl"]["fig"], 'r')
datalist = f.readlines()
f.close()

x = []
y = []
for data in datalist:
    if float(data.split(",")[0]) > 1:
        y.append(float(data.split(",")[0]))
        x.append(int(data.split(",")[1]))

# プロット
plt.ylim(0,100)
plt.scatter(x, y, label="test")

# 凡例の表示
plt.legend()
plt.grid()
# プロット表示(設定の反映)
plt.tight_layout()
plt.show()
