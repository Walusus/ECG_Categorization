import pandas as pd
import numpy as np
import data_augmentation as da
import matplotlib.pyplot as plt
from cnn_net_model import CnnNet
import sklearn.metrics as mtr
import torch
import torch.nn as nn
import seaborn as sns
import gc


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
plt.ylabel("Value")
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

# Determine train and test sets sizes
train_class_size = min(sum(y_train_labels == 0), sum(y_train_labels == 1), sum(y_train_labels == 2),
                       sum(y_train_labels == 3), sum(y_train_labels == 4))
test_class_size = min(sum(y_test_labels == 0), sum(y_test_labels == 1), sum(y_test_labels == 2),
                      sum(y_test_labels == 3), sum(y_test_labels == 4))

# Delete redundant data
train_ind_list = np.random.choice(np.argwhere(y_train_labels == 0).squeeze(), size=train_class_size+1000)  # 1000 extra samples of the hardest class
test_ind_list = np.random.choice(np.argwhere(y_test_labels == 0).squeeze(), size=test_class_size)

for i in range(1, 5):
    train_ind_list = np.hstack((train_ind_list, np.random.choice(np.argwhere(y_train_labels == i).squeeze(),
                                                                 size=train_class_size)))
    test_ind_list = np.hstack((test_ind_list, np.random.choice(np.argwhere(y_test_labels == i).squeeze(),
                                                               size=test_class_size)))

x_train = x_train[train_ind_list]
y_train_labels = y_train_labels[train_ind_list]
x_test = x_test[test_ind_list]
y_test_labels = y_test_labels[test_ind_list]

# Cleanup
del gen_data, gen_data_size, random_order, train_ind_list


# Reshape y vector to m x number of classes.
y_train = np.zeros((y_train_labels.shape[0], 5))
y_test = np.zeros((y_test_labels.shape[0], 5))

for i in range(5):
    y_train[np.argwhere(y_train_labels == i), i] = 1
    y_test[np.argwhere(y_test_labels == i), i] = 1

# Cleanup
del y_train_labels, y_test_labels


# Wrap data with tensors and create dataloader for training and test set.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x_train_tensor = torch.tensor(x_train, dtype=torch.float64, device=device).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float64, device=device)
x_test_tensor = torch.tensor(x_test, dtype=torch.float64, device=device).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float64, device=device)

batch_size = 500
# noinspection PyUnresolvedReferences
train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
# noinspection PyUnresolvedReferences
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Build network model, choose optimizer and loss function.
net = CnnNet().to(device=device, dtype=torch.float64)
# Load model weights
ans = input("Load model? [y/n]")
if ans is 'y':
    filename = input("File name: ")
    net.load_state_dict(torch.load("weights/" + filename + ".pt"))
    net.train()

learning_rate = 5e-4
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()


# Network training.
test_loss_track = []
train_loss_track = []
epochs_number = 75
for epoch in range(epochs_number):
    for batch_num, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss_track.append(loss.item())

        # Printing log per batch
        print(f"Epoch: {epoch+1:d},\tMini-batch: {batch_num+1:d},\tLoss: {loss.item():.4f}")

        # Cleanup
        del inputs, labels, outputs, loss
        gc.collect()

    # Printing summary every epoch
    test_outputs = net(x_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    test_accuracy = mtr.accuracy_score(torch.max(y_test_tensor, 1)[1].cpu(), torch.max(test_outputs, 1)[1].cpu())
    print(f"Test loss: {test_loss.item():.4f},\tTest accuracy: {test_accuracy:.2f}")
    test_loss_track.append(test_loss.item())

    # Cleanup
    del test_outputs, test_loss
    gc.collect()


# Plot learning results.
plt.figure(figsize=(8, 4))
plt.subplot(111)
plt.plot(np.linspace(1, epochs_number, num=len(train_loss_track)), train_loss_track, label="Train loss")
plt.plot(np.linspace(1, epochs_number, num=len(test_loss_track)), test_loss_track, label="Test loss")
plt.title(f"Lr={learning_rate:.5f}, Batch_size={batch_size:d}")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.show()


# Display metrics and confusion matrix.
test_outputs = net(x_test_tensor)
true_labels, pred_labels = torch.max(y_test_tensor, 1)[1].cpu(), torch.max(test_outputs, 1)[1].cpu()

accuracy = mtr.accuracy_score(true_labels, pred_labels)
class_report = mtr.classification_report(true_labels, pred_labels)
conf_mat = mtr.confusion_matrix(true_labels, pred_labels)
axis_sum = conf_mat.sum(axis=1)

print(f"\nTest set accuracy: {accuracy:.2f}")
print(class_report)

# Plot confusion matrix
class_labels = ['N', 'S', 'V', 'F', 'Q']
plt.figure(figsize=(6, 6))
sns.heatmap(conf_mat, cmap="YlGnBu", annot=True, cbar=False, square=True,
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.title(f"Test set accuracy: {accuracy:.2f}")
plt.show()

# Save model weights
ans = input("Save model? [y/n]")
if ans is 'y':
    filename = input("Save as: ")
    torch.save(net.state_dict(), "weights/" + filename + ".pt")
