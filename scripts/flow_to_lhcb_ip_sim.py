#!/usr/bin/env python3
import torch
import os
import utils


def plot_path(sfx):
    return f"plots/first_flow__{sfx}.png"


def main():

    if not os.path.exists("plots"):
        os.makedirs("plots")

    print("INFO:\tLoading target data...")
    n_samples = 10000
    target_sample = "Z"
    print(f"INFO:\tWill use {n_samples} samples throughout...")
    target_data = utils.load_data(target_sample, n_samples)
    utils.plot_from_sample(target_data, plot_path("target_sample"))

    torch.manual_seed(42)
    print("INFO:\tInstantiating spline flow...")
    quad_flow = utils.make_1d_quad_flow()
    with torch.inference_mode():
        utils.plot_from_sample(
            utils.sample_flow(quad_flow, n_samples), plot_path("flow_sample_pretrain"))

    print("INFO:\tTraining flow...")
    utils.train_flow(quad_flow, target_data, n_iter=10000, plot_path=plot_path("training"))
    with torch.inference_mode():
        utils.plot_from_sample(
            utils.sample_flow(quad_flow, n_samples), plot_path("flow_sample_posttrain"))

    with torch.inference_mode():
        utils.benchmark_hep_style(quad_flow, target_data, plot_path=plot_path("performance_plots"))

    print("INFO:\tFlow training complete. Please find plots under plots/")


if __name__ == "__main__":
    main()
