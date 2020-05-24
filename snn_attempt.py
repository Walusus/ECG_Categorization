import os
import sys

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from bindsnet.utils import get_square_weights, get_square_assignments
from torchvision import transforms
from tqdm import tqdm
from time import time as t
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder, poisson
from bindsnet.network.monitors import Monitor
from bindsnet.models import DiehlAndCook2015
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from numpy import genfromtxt
import dask.dataframe as dd
import pandas as pd
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_weights,
    plot_assignments,
    plot_performance,
    plot_voltages,
)
# hyper parameters for SNN
seed = 0
n_neurons = 100
n_epochs = 1
n_train = 1000
n_clamp = 1
n_test = 5000
n_workers = -1
theta_plus = 0.05
time = 250
dt = 1.0
intensity = 64
progress_interval = 10
plot_interval = 250
update_interval = 250
plot = True
gpu = False
exc = 22.5
inh = 120
update_inhibation_weights = 500
batch_size = 1
device = "cpu"
n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity
torch.manual_seed(seed)


# Build network.
# We use DiehlCook2015 network becaus according to bindsnet paper,
# it gave the best results for MNIST supervised learning
per_class = int(n_neurons / 5)

network = DiehlAndCook2015(
    n_inpt=187,
    exc=exc,
    inh=inh,
    dt=dt,
    nu=[1e-2, 1e-4],
    norm=78.4,
    theta_plus=theta_plus,
    inpt_shape=(1, 187),
)
network.to(device)

#Adding monitors to save voltage states on exictatory layer and inhibitatory layer
exc_voltage_monitor = Monitor(network.layers["Ae"], ["v"], time=time)
inh_voltage_monitor = Monitor(network.layers["Ai"], ["v"], time=time)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Pulling training dataset
source = pd.read_csv("heartbeat/mitbih_train.csv")
target = pd.read_csv("heartbeat/mitbih_train.csv", usecols=[187])
source.drop(source.columns[len(source.columns)-1], axis=1, inplace=True)
source_tensor = torch.tensor(source.values, dtype=torch.float)
target_tensor = torch.tensor(target.values, dtype=torch.float)
dataset = torch.utils.data.TensorDataset(source_tensor, target_tensor)

#Create data loader
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

#Defining Poisson Encoder which converts binary floats into spike
#trains of 250ms duration each
encoder = PoissonEncoder(dt=dt, time=time)

#Tensor which stores spikes for every layer
spike_record = torch.zeros(update_interval, time, n_neurons)

#Defining tensors and number of classes
#assigments -  stores labels assigned according to firing rate, for every group of neurons (We have 5 groups of "output" neurons each group  has 20 neurons.
# ex: Group 1 fires most when input belongs to class nr 2 then group 1 gets 2 as assignment)
#proportions - weights assigments for every group according to how wrong the group was in past
#rates - stores firing rates number_spikes/time

n_classes = 5
assignments = - torch.zeros(n_neurons).cpu()
proportions = torch.zeros(n_neurons, n_classes).cpu()
rates = torch.zeros(n_neurons, n_classes).cpu()
accuracy = {"all": [], "proportion": []}

#Defining monitors to monitor spike
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

labels = torch.empty(update_interval)

#save locations for plots
inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None
assigns_im = None
save_weights_fn = "plots/weights/weights.png"
save_performance_fn = "plots/performance/performance.png"
save_assaiments_fn = "plots/assaiments/assaiments.png"

directorys = ["plots", "plots/weights", "plots/performance", "plots/assaiments"]
for directory in directorys:
    if not os.path.exists(directory):
        os.makedirs(directory)

weights_mask = (1 - torch.diag(torch.ones(n_neurons))).to(device)

#Training.

print("\nStart training.\n")
n_train = 1000  # len(source_tensor)
start = t()
i = 0
for epoch in range(n_epochs):
    for X, Y in loader:
        if i > n_train:
            break
        X_enc = encoder(X*intensity)#Scaling input data and the encoding it
        torch.set_printoptions(edgeitems=50)
        label = torch.squeeze(Y, 1)
        if i % update_interval == 0 and i > 0:
            # Get network predictions.
            all_activity_pred = all_activity(spike_record, assignments, 5)
            proportion_pred = proportion_weighting(spike_record, assignments, proportions, 5)

            # Compute network accuracy according to available classification strategies.
            accuracy["all"].append(
                100 * torch.sum(labels.long() == all_activity_pred).item() / update_interval
            )
            accuracy["proportion"].append(
                100 * torch.sum(labels.long() == proportion_pred).item() / update_interval
            )

            print(
                "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                % (accuracy["all"][-1], np.mean(accuracy["all"]), np.max(accuracy["all"]))
            )
            print(
                "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f (best)\n"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )
            assignments, proportions, rates = assign_labels(spike_record, labels, 5, rates)# assigns label to neuron group according its firing rate
        labels[i % update_interval] = label[0] #labels stored for update_interval iterations
        clamp = {"Ae": per_class * label.long() + torch.Tensor(np.random.choice(int(n_neurons / 5), size=n_clamp, replace=False)).long()}
        inputs = {"X": X_enc.view(time, 1, 1, 187)}# we need to reformat input
        network.run(inputs=inputs, time=time, clamp=clamp)
        # Get voltage recording.
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")
        # Add to spikes recording.
        spike_record[i % update_interval] = spikes["Ae"].get("s").view(time, n_neurons)#Spikes for one forward pass through network stored in spike recorder
        network.reset_state_variables()  # Reset state variables.
        if i % 10 == 0:#print complete percentage
            sys.stdout.write('Completed %2f%% of train dataset. \r' % float(100*i/n_train))
            sys.stdout.flush()
        i+=1
    print("Epoch number: "+str(epoch)+" has ended.\n")


print("Testing....\n")

#Pulling test data set
source = pd.read_csv("heartbeat/mitbih_test.csv")
target = pd.read_csv("heartbeat/mitbih_test.csv", usecols=[187])
source.drop(source.columns[len(source.columns)-1], axis=1, inplace=True)
source_tensor = torch.tensor(source.values, dtype=torch.float)
target_tensor = torch.tensor(target.values, dtype=torch.float)
encoder = PoissonEncoder(dt=dt, time=time)
source_tensor = source_tensor*intensity
dataset = torch.utils.data.TensorDataset(source_tensor, target_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
#Spike recorder to store spikes
spike_record = torch.zeros(1, time, n_neurons)
hit = 0
i = 0
for X, Y in loader:
    X_enc = encoder(X)
    Y = torch.squeeze(Y, 1)
    inputs = {"X": X_enc.view(time, 1, 1, 187)}
    network.run(inputs=inputs, time=time)
    out_spikes = network.monitors["Ae_spikes"].get("s").view(time, n_neurons)#Recorded spikes for one forward pass
    spike_record[0] = out_spikes
    class_spike = torch.zeros(5)
    for c in range(5):
        class_spike[c] = out_spikes[c * 20 : (c + 1) * 20].sum()
    maxInd = class_spike.argmax()
    out_spikes = out_spikes.squeeze().sum(dim=0) #summing which label had the most spikes
    all_activity_pred = all_activity(spike_record, assignments, 5) #using activity prediction from tarining see if it changes anything
    print("Output Sum: "+str(maxInd)+" Activ. Pred: "+str(all_activity_pred[0])+" Correct: "+str(Y[0]))
    print(all_activity_pred)
    if maxInd == Y[0]:
        hit += 1
    if i % 10:
        print("Accuracy: "+str(hit/(i+1)))
    i += 1
acc = hit / n_test
print("\n Accuracy: " + str(acc) + "\n")
