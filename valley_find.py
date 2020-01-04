import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Generate a noisy AR(1) sample
# np.random.seed(0)
# rs = np.random.randn(200)
# xs = [0]
# for r in rs:
#     xs.append(xs[-1]*0.9 + r)
# df = pd.DataFrame(xs, columns=['data'])

df=pd.read_csv("D:/Research/codes copy/smoothen_y.csv")
print(df)
# print(df.type())

# Find local peaks
df['min'] = df.data[(df.data.shift(1) > df.data) & (df.data.shift(-1) > df.data)]
df['max'] = df.data[(df.data.shift(1) < df.data) & (df.data.shift(-1) < df.data)]

# # Plot results
# X=df.Z
# Y=df.data
plt.scatter(df.Z,df['min'], c='r')
# plt.scatter(df.Z, df['max'], c='g')
plt.plot(df.Z,df.data)
plt.ylim(-5,35)
plt.show()


