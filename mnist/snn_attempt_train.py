import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time as t
from bindsnet.encoding import poisson
from bindsnet.network import load
from bindsnet.network.monitors import Monitor
from bindsnet.models import DiehlAndCook2015v2
from bindsnet.evaluation import assign_labels, update_ngram_scores
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.analysis.plotting import plot_input, plot_spikes, plot_weights, plot_assignments, plot_performance
from mnist.utils import update_curves, print_results


def main():
    seed = 0            #random seed
    n_neurons = 100     # number of neurons per layer
    n_train = 60000     # number of traning examples to go through
    inhib = 100.0       # strength of synapses from inh. layer to exci. layer
    lr = 1e-2               # learning rate
    lr_decay = 0.99         # learning rate decay
    time = 350              # duration of each sample after running through possion encoder
    dt = 1                  # timestep
    theta_plus = 0.05       # post spike threshold increase
    tc_theta_decay = 1e7    # threshold decay
    intensity = 0.5         # number to multiply input
    progress_interval = 10
    update_interval = 250
    plot = False
    gpu = False
    load_network = False   # load network from disk
    n_classes = 10
    n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
    # TRAINING
    assert n_train % update_interval == 0
    np.random.seed(seed)

    if gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    # Build network
    if load_network:
        network = load('net3.pt') # here goes file with network to load
    else:
        network = DiehlAndCook2015v2(
                n_inpt=784, n_neurons=n_neurons, inh=inhib, dt=dt, norm=78.4,
                theta_plus=theta_plus, tc_theta_decay=tc_theta_decay, nu=[0, lr]
            )
    if gpu:
        network.to("cuda")
    n_epochs = 2
    # Pull dataset
    data, targets = torch.load('data/MNIST/TorchvisionDatasetWrapper/processed/training.pt')
    data = data * intensity
    trainset = torch.utils.data.TensorDataset( data ,targets )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,shuffle=False, num_workers=1)


    # Spike recording
    spike_record = torch.zeros(update_interval, time, n_neurons)
    full_spike_record = torch.zeros(n_train, n_neurons).long()


    # Intialization
    if load_network:
        assignments, proportions, rates, ngram_scores = torch.load('params3.pt')
    else:
        assignments = -torch.ones_like(torch.Tensor(n_neurons))
        proportions = torch.zeros_like(torch.Tensor(n_neurons, n_classes))
        rates = torch.zeros_like(torch.Tensor(n_neurons, n_classes))
        ngram_scores = {}
    curves = {'all': [], 'proportion': [], 'ngram': []}
    predictions = {
        scheme: torch.Tensor().long() for scheme in curves.keys()
    }
    best_accuracy = 0

    # Initilize spike records
    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=['s'], time=time)
        network.add_monitor(spikes[layer], name='%s_spikes' % layer)
    i = 0
    current_labels = torch.zeros(update_interval)
    inpt_axes = None
    inpt_ims = None
    spike_ims = None
    spike_axes = None
    weights_im = None
    assigns_im = None
    perf_ax = None
    # train
    train_time = t.time()


    time1 = t.time()
    for j in range(n_epochs):
        i = 0
        for sample, label in trainloader:
            sample = sample.view(784)
            if i >= n_train:
                break
            if i % progress_interval == 0:
                print(f'Progress: {i} / {n_train} took {(t.time()-time1)} s')
                time1 = t.time()
            if i % update_interval == 0 and i > 0:
                network.connections['X','Y'].update_rule.nu[1] *= lr_decay
                # update accuracy evaluation
                curves, preds = update_curves(
                    curves, current_labels, n_classes, spike_record=spike_record, assignments=assignments,
                    proportions=proportions, ngram_scores=ngram_scores, n=2
                )
                print_results(curves)
                for scheme in preds:
                    predictions[scheme] = torch.cat([predictions[scheme], preds[scheme]], -1)
                # Accuracy curves
                if any([x[-1] > best_accuracy for x in curves.values()]):
                    print('New best accuracy! Saving network parameters to disk.')

                    # Save network and parameters to disk.
                    network.save( os.path.join('net_output.pt'))
                    path = "parameters_output.pt"
                    torch.save((assignments, proportions, rates, ngram_scores), open(path, 'wb'))
                    best_accuracy = max([x[-1] for x in curves.values()])
                assignments, proportions, rates = assign_labels(spike_record, current_labels, n_classes, rates)
                ngram_scores = update_ngram_scores(spike_record, current_labels, n_classes, 2, ngram_scores)
            sample_enc = poisson(datum=sample,time=time,dt=dt)
            inpts = {'X':sample_enc }
            # Run the network on the input.
            network.run(inputs=inpts, time=time)
            retries = 0
            while spikes['Y'].get('s').sum() < 1 and retries < 3:
                retries += 1
                sample =sample*2
                inpts = {'X' : poisson(datum=sample, time=time,dt=dt)}
                network.run(inputs=inpts, time=time)

            # Spikes reocrding
            spike_record[i % update_interval] = spikes['Y'].get('s').view(time,n_neurons)
            full_spike_record[i] = spikes['Y'].get('s').view(time,n_neurons).sum(0).long()
            if plot:
                _input = sample.view(28, 28)
                reconstruction = inpts['X'].view(time, 784).sum(0).view(28, 28)
                _spikes = {layer: spikes[layer].get('s') for layer in spikes}
                input_exc_weights = network.connections[('X', 'Y')].w
                square_weights = get_square_weights(input_exc_weights.view(784, n_neurons), n_sqrt, 28)
                square_assignments = get_square_assignments(assignments, n_sqrt)
                inpt_axes, inpt_ims = plot_input(_input, reconstruction, label=label, axes=inpt_axes, ims=inpt_ims)
                spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
                weights_im = plot_weights(square_weights, im=weights_im)
                assigns_im = plot_assignments(square_assignments, im=assigns_im)
                perf_ax = plot_performance(curves, ax=perf_ax)
                plt.pause(1e-8)
            current_labels[i % update_interval] = label[0]
            network.reset_state_variables()
            i += 1
        print(f'Number of epochs {j}/{n_epochs+1}')
        torch.save(network.state_dict(),'net_final.pt')
        path = "parameters_final.pt"
        torch.save((assignments, proportions, rates, ngram_scores), open(path, 'wb'))
    print("Training completed. Training took "+str((t.time()-train_time)/6)+" min.")
    print("Saving network...")
    network.save(os.path.join('net_final.pt'))
    torch.save((assignments, proportions, rates, ngram_scores), open('parameters_final.pt', 'wb'))
    print("Network saved.")



if __name__ == '__main__':
    main()
