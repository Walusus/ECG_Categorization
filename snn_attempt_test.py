import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time as t
from sklearn.metrics import confusion_matrix
from bindsnet.encoding import poisson
from bindsnet.network import load
from bindsnet.network.monitors import Monitor
from bindsnet.evaluation import assign_labels, update_ngram_scores, proportion_weighting, ngram
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.analysis.plotting import plot_input, plot_spikes, plot_weights, plot_assignments, plot_performance
from utils import update_curves, print_results


def main():
    #TEST

    # hyperparameters
    n_neurons = 100
    n_test = 10000
    inhib = 100
    time = 350
    dt = 1
    intensity = 0.25
    # extra args
    progress_interval = 10
    update_interval = 250
    plot = True
    seed = 0
    train = True
    gpu = False
    n_classes = 10
    n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
    # TESTING
    assert n_test % update_interval == 0
    np.random.seed(seed)
    save_weights_fn = "plots_snn/weights/weights_test.png"
    save_performance_fn = "plots_snn/performance/performance_test.png"
    save_assaiments_fn = "plots_snn/assaiments/assaiments_test.png"
    # load network
    network =load('net_output.pt') # here goes file with network to load
    network.train(False)

    # pull dataset
    data, targets = torch.load('data/MNIST/TorchvisionDatasetWrapper/processed/test.pt')
    data =data * intensity
    data_stretched = data.view(len(data), -1, 784)
    testset = torch.utils.data.TensorDataset(data_stretched, targets)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    # spike init
    spike_record = torch.zeros(update_interval, time, n_neurons)
    full_spike_record = torch.zeros(n_test, n_neurons).long()
    # load parameters
    assignments, proportions,rates, ngram_scores = torch.load('parameters_output.pt')# here goes file with parameters to load
    # accuracy initialization
    curves = {'all': [], 'proportion': [], 'ngram': []}
    predictions = {
        scheme: torch.Tensor().long() for scheme in curves.keys()
    }
    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=['s'], time=time)
        network.add_monitor(spikes[layer], name='%s_spikes' % layer)
    print("Begin test.")
    inpt_axes = None
    inpt_ims = None
    spike_ims = None
    spike_axes = None
    weights_im = None
    assigns_im = None
    perf_ax = None
    i = 0
    current_labels = torch.zeros(update_interval)

    # test
    test_time = t.time()
    time1 = t.time()
    for sample, label in testloader:
        sample = sample.view(1,1, 28,28)
        if i % progress_interval == 0:
            print(f'Progress: {i} / {n_test} took {(t.time()-time1)*10000} s')
        if i % update_interval == 0 and i > 0:
            # update accuracy evaluation
            curves, preds = update_curves(
                curves, current_labels, n_classes, spike_record=spike_record, assignments=assignments,
                proportions=proportions, ngram_scores=ngram_scores, n=2
            )
            print_results(curves)
            for scheme in preds:
                predictions[scheme] = torch.cat([predictions[scheme], preds[scheme]], -1)
        sample_enc = poisson(datum=sample,time=time,dt=dt)
        inpts = {'X':sample_enc }
        # Run the network on the input.
        network.run(inputs=inpts, time=time)
        retries = 0
        while spikes['Ae'].get('s').sum() < 1 and retries < 3:
            retries += 1
            sample =sample*2
            inpts = {'X' : poisson(datum=sample, time=time,dt=dt)}
            network.run(inputs=inpts, time=time)

        # Spikes reocrding
        spike_record[i % update_interval] = spikes['Ae'].get('s').view(time,n_neurons)
        full_spike_record[i] = spikes['Ae'].get('s').view(time,n_neurons).sum(0).long()
        if plot:
            _input = sample.view(28, 28)
            reconstruction = inpts['X'].view(time, 784).sum(0).view(28, 28)
            _spikes = {layer: spikes[layer].get('s') for layer in spikes}
            input_exc_weights = network.connections[('X', 'Ae')].w
            square_assignments = get_square_assignments(assignments, n_sqrt)
            assigns_im = plot_assignments(square_assignments, im=assigns_im)
            if i % update_interval == 0:  # plot weights on every update interval
                square_weights = get_square_weights(input_exc_weights.view(784, n_neurons), n_sqrt, 28)
                weights_im = plot_weights(square_weights, im=weights_im)
                [weights_im, save_weights_fn] = plot_weights(square_weights, im=weights_im, save=save_weights_fn)
            inpt_axes, inpt_ims = plot_input(_input, reconstruction, label=label, axes=inpt_axes, ims=inpt_ims)
            spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
            assigns_im = plot_assignments(square_assignments, im=assigns_im, save=save_assaiments_fn)
            perf_ax = plot_performance(curves, ax=perf_ax, save=save_performance_fn)
            plt.pause(1e-8)
        current_labels[i % update_interval] = label[0]
        network.reset_state_variables()
        if i % 10 == 0 and i > 0:
            preds = ngram(spike_record[i%update_interval-10:i%update_interval], ngram_scores, n_classes, 2)
            print(f'Predictions: {(preds*1.0).numpy()}')
            print(f'True value:  {current_labels[i%update_interval-10:i%update_interval].numpy()}')
        time1 = t.time()
        i += 1
        # Compute confusion matrices and save them to disk.
        confusions = {}
    for scheme in predictions:
        confusions[scheme] = confusion_matrix(targets, predictions[scheme])
        to_write = 'confusion_test'
        f = '_'.join([str(x) for x in to_write]) + '.pt'
        torch.save(confusions, os.path.join('.', f))
    print("Test completed. Testing took "+str((t.time()-test_time)/6)+" min.")




if __name__ == '__main__':
    main()
