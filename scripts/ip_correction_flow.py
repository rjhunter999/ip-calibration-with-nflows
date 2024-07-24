#!/usr/bin/env python3
import torch
import os
import utils


def plot_path(sample_type, sfx):
    return f"plots/spline_flow__{sample_type}__{sfx}.png"


def main():

    if not os.path.exists("plots"):
        os.makedirs("plots")

    n_candidates = 1000
    # Train two flows, one to model the LHCb data, and one to model the LHCb simulation
    for target_sample in ('Z', 'DATA'):
        print(f"INFO:\tTraining flow for {target_sample}...")
        print('-'*80)

        print("INFO:\tLoading target data...")
        print(f"INFO:\tWill use {n_candidates} candidates throughout...")
        target_data = utils.load_data(target_sample, n_candidates)
        utils.plot_from_sample(target_data, plot_path(target_sample, "target_sample"))

        torch.manual_seed(42)
        print("INFO:\tInstantiating spline flow...")
        quad_flow = utils.make_1d_quad_flow()
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

        print(f"INFO:\tFlow training complete for target sample '{target_sample}'. Please find plots under plots/")


if __name__ == "__main__":
    main()
