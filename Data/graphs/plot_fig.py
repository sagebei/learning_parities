import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

# font = {'family': 'normal',
#         'weight': 'small',
#         'size': 14}
#
# matplotlib.rc('font', **font)

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


fig, ax = plt.subplots()
right_side = ax.spines["right"]
upper_side = ax.spines['top']
right_side.set_visible(False)
upper_side.set_visible(False)

alphazero_file_path = f'run-lstm30_30_1_200_10000_100000_False-0-0.0_aug-tag-train_batch_accuracy.csv'
data = pd.read_csv(alphazero_file_path)
x_alphazero = data['Step']
y_alphazero = data['Value']
y_alphazero = smooth(y_alphazero, 0.3)

random_file_path = f'run-lstm30_30_1_200_10000_100000_False-0-0.3_aug-tag-train_batch_accuracy.csv'
data = pd.read_csv(random_file_path)
print(data.shape)
x_random = data['Step']
y_random = data['Value']
# y_random = smooth(y_random, 0.6)

file_path = f'run-lstm30_30_1_200_10000_100000_False-0-0.0_uniform-tag-train_batch_accuracy.csv'
data = pd.read_csv(file_path)
print(data.shape)
x_uniform = data['Step']
y_uniform = data['Value']

plt.plot(x_random, y_random, color='royalblue', linewidth=1.5, label='Normal Approach with NDA')
plt.plot(x_uniform, y_uniform, color='g', linewidth=1.5, label='Uniform Approach')
plt.plot(x_alphazero, y_alphazero, color='dimgray', linewidth=1.5, label='Normal Approach')


plt.xlim([0, 145000])
# plt.ylim([min(y_alphazero) - 0.05, max(y_alphazero) + 0.05])

plt.ylabel(f'Training Batch Accuracy')
plt.xlabel('Training steps (n)')
plt.grid(True)
leg = plt.legend(loc='best', frameon=False)
for line in leg.get_lines():
    line.set_linewidth(3)

# plt.tight_layout()
plt.savefig(f'nda.jpg')
plt.show()


