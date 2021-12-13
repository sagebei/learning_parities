from parityfunction.Data.utils import ParityDataset
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

n_samples = 50000
n_elems = 20

approach1 = ParityDataset(n_samples=n_samples,
                          n_elems=n_elems,
                          n_nonzero_min=0,
                          n_nonzero_max=n_elems,
                          exclude_dataset=None,
                          unique=False,
                          model='mlp',
                          approach=1,
                          data_augmentation=0)

approach2 = ParityDataset(n_samples=n_samples,
                          n_elems=n_elems,
                          n_nonzero_min=0,
                          n_nonzero_max=n_elems,
                          exclude_dataset=None,
                          unique=False,
                          model='mlp',
                          approach=2,
                          data_augmentation=0)

approach3 = ParityDataset(n_samples=n_samples,
                          n_elems=n_elems,
                          n_nonzero_min=0,
                          n_nonzero_max=n_elems,
                          exclude_dataset=None,
                          unique=False,
                          model='mlp',
                          approach=3,
                          data_augmentation=0.3)

data1 = {i: 0 for i in range(n_elems+1)}
for i, j in Counter((approach1.X == 1).sum(dim=1).numpy()).items():
    data1[i] = j
data1 = sorted(data1.items(), key=lambda i: i[0])

data2 = {i: 0 for i in range(n_elems+1)}
for i, j in Counter((approach2.X == 1).sum(dim=1).numpy()).items():
    data2[i] = j
data2 = sorted(data2.items(), key=lambda i: i[0])

data3 = {i: 0 for i in range(n_elems+1)}
for i, j in Counter((approach3.X == 1).sum(dim=1).numpy()).items():
    data3[i] = j
data3 = sorted(data3.items(), key=lambda i: i[0])

print(data1)
print(data2)
print(data3)

plt.plot([i[0] for i in data1], [i[1] for i in data1], color='b', linewidth=1.5, label='Approach 1')
plt.plot([i[0] for i in data2], [i[1] for i in data2], color='y', linewidth=1.5, label='Approach 2')
plt.plot([i[0] for i in data3], [i[1] for i in data3], color='g', linewidth=1.5, label='Approach 3')

plt.ylabel(f'number of data samples')
plt.xlabel('number of 1s')
plt.grid(True)
plt.xticks(np.arange(0, n_elems+1, step=4))
leg = plt.legend(loc='best', frameon=False)
for line in leg.get_lines():
    line.set_linewidth(3)

plt.savefig(f'data_generation.jpg')
plt.show()




