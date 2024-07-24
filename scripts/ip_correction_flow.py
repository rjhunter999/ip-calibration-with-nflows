#!/usr/bin/env python3
import torch
import os
import utils
import numpy as np
from nflows import transforms
import matplotlib.pyplot as plt


def plot_path(sample_type, sfx):
    return f"plots/spline_flow__{sample_type}__{sfx}.png"


def plot_single_transform(target_sample, transform):

    bin_opts = dict(bins=60, range=(-4, 2))
    target_data = utils.load_data(target_sample, 1000)
    target_hist, bins = np.histogram(target_data, **bin_opts)

    noise = torch.randn(1000, 1)
    transformed = transforms.InverseTransform(transform)(noise)

    _, ax = plt.subplots()
    bin_centres = (bins[1:] + bins[:-1]) / 2
    ax.errorbar(bin_centres, target_hist, yerr=np.sqrt(target_hist), fmt='+', label=f"{target_sample} target", color='black')
    ax.hist(noise.numpy(), bins, histtype="stepfilled", label='Init. noise', facecolor='blue', alpha=0.3)
    ax.hist(transformed[0].numpy(), bins, histtype="stepfilled", label='Transformed', facecolor='red', alpha=0.3)
    ax.legend()
    plt.savefig(plot_path(target_sample, "transform_validation"))


def main():

    if not os.path.exists("plots"):
        os.makedirs("plots")

    n_candidates = 1000
    torch.manual_seed(42)
    # Train two flows, one to model the LHCb data, and one to model the LHCb simulation
    transforms = {}
    for target_sample in ('Z', 'DATA'):
        print(f"INFO:\tTraining flow for {target_sample}...")
        print('-'*80)

        print("INFO:\tLoading target data...")
        print(f"INFO:\tWill use {n_candidates} candidates throughout...")
        target_data = utils.load_data(target_sample, n_candidates)
        utils.plot_from_sample(target_data, plot_path(target_sample, "target_sample"))

        print("INFO:\tInstantiating spline flow...")
        transform, quad_flow = utils.make_1d_quad_flow()
        with torch.inference_mode():
            utils.plot_from_sample(
                utils.sample_flow(quad_flow, n_candidates), plot_path(target_sample, "pretrain"))

        print("INFO:\tTraining flow...")
        utils.train_flow(quad_flow, target_data, n_iter=1000, plot_path=plot_path(target_sample, "training"))
        with torch.inference_mode():
            utils.plot_from_sample(
                utils.sample_flow(quad_flow, n_candidates), plot_path(target_sample, "posttrain"))

        with torch.inference_mode():
            utils.benchmark_hep_style(quad_flow, target_data, plot_path=plot_path(target_sample, "performance_plots"))

        transforms[target_sample] = transform
        print(f"INFO:\tFlow training complete for target sample '{target_sample}'. Please find plots under plots/")

    # Make sure each transform individually is working
    for target_sample, transform in transforms.items():
        with torch.inference_mode():
            plot_single_transform(target_sample, transform)


if __name__ == "__main__":
    main()
