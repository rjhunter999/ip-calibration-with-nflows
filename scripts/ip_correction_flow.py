#!/usr/bin/env python3
import torch
import os
import utils
import numpy as np
from nflows import transforms
import matplotlib.pyplot as plt
import argparse


def plot_path(sample_type, sfx):
    return f"plots/spline_flow__{sample_type}__{sfx}.png"


def plot_single_transform(target_sample: torch.Tensor, init_sample: torch.Tensor, transform: transforms.Transform, plot_path: str, nbins=60, xrange=(-4, 2)):

    bin_opts = dict(bins=nbins, range=xrange)
    target_hist, bins = np.histogram(target_sample, **bin_opts)

    transformed = transform(init_sample)

    _, ax = plt.subplots()
    bin_centres = (bins[1:] + bins[:-1]) / 2
    ax.errorbar(bin_centres, target_hist, yerr=np.sqrt(target_hist), fmt='+', label="Target", color='black')
    ax.hist(init_sample.numpy(), bins, histtype="stepfilled", label='Initial', facecolor='blue', alpha=0.3)
    ax.hist(transformed[0].numpy(), bins, histtype="stepfilled", label='Transformed', facecolor='red', alpha=0.3)
    ax.legend()
    plt.savefig(plot_path)


def main():
    parser = argparse.ArgumentParser(description="Train a flow to correct the LHCb IP simulation to match the data.")
    parser.add_argument("--test", action="store_true", help="Run with smaller datasets and less training iterations for testing purposes.")
    args = parser.parse_args()

    if args.test:
        n_candidates = 100
        n_iter = 100
    else:
        n_candidates = 100000
        n_iter = 10000

    if not os.path.exists("plots"):
        os.makedirs("plots")

    datasets = {target_sample: utils.load_data(target_sample, n_candidates) for target_sample in ('Z', 'DATA')}

    # Train two flows, one to model the LHCb data, and one to model the LHCb simulation
    torch.manual_seed(42)
    trained_transforms = {}
    for target_sample in ('Z', 'DATA'):
        print(f"INFO:\tTraining flow for {target_sample}...")
        print('-'*80)

        print("INFO:\tLoading target data...")
        print(f"INFO:\tWill use {n_candidates} candidates throughout...")
        target_data = datasets[target_sample]
        utils.plot_from_sample(target_data, plot_path(target_sample, "target_sample"))

        print("INFO:\tInstantiating spline flow...")
        transform, quad_flow = utils.make_1d_quad_flow()
        with torch.inference_mode():
            utils.plot_from_sample(
                utils.sample_flow(quad_flow, n_candidates), plot_path(target_sample, "pretrain"))

        print("INFO:\tTraining flow...")
        # Generally requires 10k iterations
        utils.train_flow(quad_flow, target_data, n_iter=n_iter, plot_path=plot_path(target_sample, "training"))
        with torch.inference_mode():
            utils.plot_from_sample(
                utils.sample_flow(quad_flow, n_candidates), plot_path(target_sample, "posttrain"))

        with torch.inference_mode():
            utils.benchmark_hep_style(quad_flow, target_data, plot_path=plot_path(target_sample, "performance_plots"))

        trained_transforms[target_sample] = transform
        print(f"INFO:\tFlow training complete for target sample '{target_sample}'")

    # Make sure each transform individually is working
    for target_sample, transform in trained_transforms.items():
        with torch.inference_mode():
            init_noise = torch.randn(n_candidates, 1)
            # Remember that the transform is from the target -> noise
            plot_single_transform(datasets[target_sample], init_noise, transforms.InverseTransform(transform), plot_path(target_sample, "transform_validation"))

    print("INFO:\tPutting the two transforms together...")
    # A forward transform goes from target -> noise
    sim_to_data_transform = transforms.CompositeTransform(
        [trained_transforms["Z"], transforms.InverseTransform(trained_transforms["DATA"])]
    )
    with torch.inference_mode():
        plot_single_transform(target_sample=datasets['DATA'], init_sample=datasets['Z'], transform=sim_to_data_transform, plot_path=plot_path("Z_to_DATA", "transform_validation"), nbins=100, xrange=(-4, 0))


if __name__ == "__main__":
    main()
