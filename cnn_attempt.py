import pandas as pd
import numpy as np
import data_augmentation as da
import matplotlib.pyplot as plt
import torch


# Load mit-bih training dataset using pandas and convert to numpy. Divide into X and y.
df_train = pd.read_csv("datasets/mitbih_train.csv", header=None)
df_test = pd.read_csv("datasets/mitbih_test.csv", header=None)

# Printing some data info
df_train.info(), df_test.info()
print("\nTest class counts:\n", df_test[187].value_counts())
print("\nTrain class counts:\n", df_train[187].value_counts())

# Converting to numpy
arr_train = df_train.to_numpy()
arr_test = df_test.to_numpy()
x_train = arr_train[:, 0:-1]
y_train_labels = arr_train[:, -1]
x_test = arr_test[:, 0:-1]
y_test_labels = arr_test[:, -1]

# Cleanup
del arr_train, arr_test
del df_train, df_test


# Extract and plot sample of each class.
c0_sample = x_test[np.argwhere(y_test_labels == 0)[0]].squeeze()
c1_sample = x_test[np.argwhere(y_test_labels == 1)[0]].squeeze()
c2_sample = x_test[np.argwhere(y_test_labels == 2)[0]].squeeze()
c3_sample = x_test[np.argwhere(y_test_labels == 3)[0]].squeeze()
c4_sample = x_test[np.argwhere(y_test_labels == 4)[0]].squeeze()

# Plotting
time_axes = np.arange(0, 1/125 * 187, 1/125)  # sampling frequency is 125Hz
plt.figure(figsize=(10, 5))
plt.subplot(111)
plt.plot(time_axes, c0_sample, label="N: Normal beat")
plt.plot(time_axes, c1_sample, label="S: Supraventricular premature beat")
plt.plot(time_axes, c2_sample, label="V: Premature ventricular contraction")
plt.plot(time_axes, c3_sample, label="F: Fusion of ventricular and normal beat")
plt.plot(time_axes, c4_sample, label="Q: Unclassifiable beat")
plt.title("Each class normalized ECG signal sample")
plt.xlabel("Time [ms]")
plt.ylabel("value")
plt.legend()
plt.show()

# Cleanup
del c0_sample, c1_sample, c2_sample, c3_sample, c4_sample, time_axes


# Augment training set (experiment with different values).
gen_data_size = 0
for i in range(3):
    # Copying and modifying class 3 samples
    gen_data = np.apply_along_axis(lambda x: da.modify_vector(x, .15), axis=1,
                                   arr=x_train[np.argwhere(y_train_labels == 3), :])
    gen_data_size += gen_data.shape[0]
    x_train = np.vstack((x_train, gen_data.squeeze()))

# Adding labels for created data
y_train_labels = np.hstack((y_train_labels, np.ones(gen_data_size) * 3))

# Shuffling dataset
random_order = np.random.permutation(len(y_train_labels))
y_train_labels = y_train_labels[random_order]
x_train = x_train[random_order]

# Determine train set size
train_set_size = min(sum(y_train_labels == 0), sum(y_train_labels == 1), sum(y_train_labels == 2),
                     sum(y_train_labels == 3), sum(y_train_labels == 4))

# Delete redundant data
ind_list = np.random.choice(np.argwhere(y_train_labels == 0).squeeze(), size=train_set_size)
ind_list = np.hstack((ind_list, np.random.choice(np.argwhere(y_train_labels == 1).squeeze(), size=train_set_size)))
ind_list = np.hstack((ind_list, np.random.choice(np.argwhere(y_train_labels == 2).squeeze(), size=train_set_size)))
ind_list = np.hstack((ind_list, np.random.choice(np.argwhere(y_train_labels == 3).squeeze(), size=train_set_size)))
ind_list = np.hstack((ind_list, np.random.choice(np.argwhere(y_train_labels == 4).squeeze(), size=train_set_size)))

x_train = x_train[ind_list]
y_train_labels = y_train_labels[ind_list]

# Cleanup
del gen_data, gen_data_size, random_order, ind_list


# Reshape y vector to m x number of classes.
y_train = np.zeros((y_train_labels.shape[0], 5))
y_test = np.zeros((y_test_labels.shape[0], 5))

for i in range(5):
    y_train[np.argwhere(y_train_labels == i), i] = 1
    y_test[np.argwhere(y_test_labels == i), i] = 1

# Cleanup
del y_train_labels, y_test_labels


# Wrap data with tensors and create dataloader for training set.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x_train_tensor = torch.tensor(x_train, dtype=torch.float64, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float64, device=device)
x_test_tensor = torch.tensor(x_test, dtype=torch.float64, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float64, device=device)

batch_size = 200
# noinspection PyUnresolvedReferences
dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
# noinspection PyUnresolvedReferences
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# TODO Build network model, choose optimizer and loss function.
pass

# TODO Network training.
pass

# TODO Display metrics and confusion matrix.
pass

# TODO Save model weights
pass
