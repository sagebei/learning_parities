import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
import numpy as np

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}

matplotlib.rc('font', **font)

n_samples = 100000
n_features = 30

data1 = np.random.randint(2, size=(n_samples, n_features))
data2 = np.random.randint(n_features, size=(n_samples,))

counter1 = Counter(data1.sum(axis=1))
counter2 = Counter(data2)

l1 = []
l2 = []
index = [i for i in range(n_features)]
for i in range(0, n_features):
    l1.append(counter1[i])
    l2.append(counter2[i])

plt.xlabel('Number of 1s')
plt.ylabel('Number of data samples')

plt.plot(index, l1, linewidth=2, label='Normal Approach')
plt.plot(index, l2, linewidth=2, label='Uniform Approach')
plt.tight_layout()
leg = plt.legend(loc='best', frameon=False)

plt.savefig('data_generation.jpg')
plt.show()