#!/usr/bin/env python3
import torch
import matplotlib.pyplot as plt
from nflows import transforms, distributions, flows
import numpy as np
import uproot


def make_1d_quad_flow():
    N_FEATURES = 1
    transform = transforms.PiecewiseRationalQuadraticCDF(
            shape=N_FEATURES, tails='linear',
            tail_bound=4, num_bins=3)
    return transform, flows.Flow(
        transform=transform,
        distribution=distributions.StandardNormal(shape=[N_FEATURES])
    )


def plot_from_sample(samples, plot_path):
    _, ax = plt.subplots()
    hist, bins = np.histogram(samples, bins=100, range=(-4, 0))
    ax.bar(x=bins[:-1], height=hist, yerr=np.sqrt(hist), width=bins[1] - bins[0])
    ax.set_xlabel('log10(IP / mm)')
    ax.set_ylabel('Candidates / bin')
    plt.savefig(plot_path)


def sample_flow(flow, n_samples):
    samples = flow.sample(n_samples)
    return samples


def train_flow(flow, target_data, n_iter, plot_path, xrange=(-4, 0)):
    # Train the flow, and periodically plot the results
    binning = dict(bins=100, range=xrange)

    # Setup the plot
    _, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax = ax.flatten()
    hist_target, bins = np.histogram(target_data, **binning)
    i_fig = 0

    optimizer = torch.optim.Adam(flow.parameters())
    for i in range(n_iter+1):
        optimizer.zero_grad()
        loss = -flow.log_prob(inputs=target_data).mean()
        loss.backward()
        optimizer.step()

        if i % (n_iter/5) != 0:
            continue

        with torch.inference_mode():
            print(f"Loss at iteration {i}: {loss}")
            ax[i_fig].set_title(f"Iteration {i}: Loss = {loss:.5f}")
            ax[i_fig].bar(x=bins[:-1], height=hist_target, yerr=np.sqrt(hist_target), width=bins[1] - bins[0], label='MC', fill=False, edgecolor='blue')
            ax[i_fig].set_xlabel('log10(IP / mm)')
            ax[i_fig].set_ylabel('Candidates / bin')

            # Sample from the flow and plot the histogram
            samples = sample_flow(flow, target_data.shape[0])
            pred_hist, bins = np.histogram(samples, **binning)
            ax[i_fig].bar(x=bins[:-1], height=pred_hist, yerr=np.sqrt(pred_hist), width=bins[1] - bins[0], label='Flow', fill=False, edgecolor='red')

            ax[i_fig].legend()
            i_fig += 1

    plt.savefig(plot_path)


def load_data(sample_type, n_samples):
    # Load up the simulated data and put it into a 1D numpy array
    file = uproot.open(f"data/tuple_for_training__{sample_type}.root")
    muon_prefix = "mup_"  # could also use mum_
    branch = f'{muon_prefix}IP'
    selection = "(1>0)"
    ip_arr = file['DecayTree'].arrays(
        branch, cut=selection, library='np')[branch].astype(np.float64)

    # take first n_samples from sim_log10_ip
    log10_ip_arr = np.log10(ip_arr)[:, :n_samples]

    # Reshape needed to swap the axes to match what the flow expects
    log10_ip_arr = torch.tensor(log10_ip_arr, dtype=torch.float32).reshape(-1, 1)
    return log10_ip_arr


def benchmark_hep_style(flow, target, plot_path, xrange=(-3.5, -0.5)):
    # Please use with torch.inference_mode

    # Divide the canvas into 2 vertically
    _, ax = plt.subplots(3, 1, figsize=(6, 12))

    bin_opts = dict(bins=50, range=xrange)
    num_hist, bins = np.histogram(target, **bin_opts)

    samples = flow.sample(target.shape[0])
    pred_hist, bins = np.histogram(samples, **bin_opts)

    # Plot the two distributions
    ax[0].bar(x=bins[:-1], height=num_hist, yerr=np.sqrt(num_hist), width=bins[1] - bins[0], label='MC', fill=False)
    ax[0].bar(x=bins[:-1], height=pred_hist, yerr=np.sqrt(pred_hist), width=bins[1] - bins[0], label='Flow', fill=False, edgecolor='red', capsize=0)
    ax[0].legend()

    # Evaluate a chi2 - treat them as two independent distributions
    resid = np.where(np.isclose(num_hist, 0), 0, (num_hist - pred_hist)**2 / (np.abs(num_hist) + np.abs(pred_hist)))
    chi2 = np.sum(resid)
    nbins = len(num_hist)
    print(f"Chi2/nbins: {chi2}/{nbins} = {chi2/nbins}")

    ratio = num_hist / pred_hist
    ax[1].axhline(y=1, color='gray', linestyle='--')
    ax[1].errorbar(x=bins[:-1], y=ratio, yerr=np.sqrt((ratio**2 / np.abs(num_hist)) + (ratio**2/np.abs(pred_hist))), xerr=bins[1] - bins[0], label='Ratio', ls='')
    ax[1].set_ylim([0.5, 1.5])
    ax[1].set_ylabel('Ratio')

    # Calculate pull histogram
    pull = (num_hist - pred_hist) / np.sqrt(np.abs(num_hist) + np.abs(pred_hist))
    ax[2].bar(x=bins[:-1], height=pull, yerr=np.zeros_like(pull), width=bins[1] - bins[0], label='Pull', fill=True, edgecolor='blue')
    ax[2].set_ylabel('Pull')
    ax[2].set_ylim([-5, 5])
    print(f"The sum of squared residuals is {np.sum(pull**2)}")
    ax[2].set_xlabel('log10(IP / mm)')

    plt.savefig(plot_path)
