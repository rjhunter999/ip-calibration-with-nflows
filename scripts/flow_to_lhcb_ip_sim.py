#!/usr/bin/env python3
import torch
import os
import utils


def main():

    if not os.path.exists("plots"):
        os.makedirs("plots")

    print("INFO:\tLoading target data...")
    n_samples = 10
    print(f"INFO:\tWill use {n_samples} samples throughout...")
    target_data = utils.load_data(n_samples)
    utils.plot_from_sample(target_data, "target_sample")

    torch.manual_seed(42)
    print("INFO:\tInstantiating spline flow...")
    quad_flow = utils.make_1d_quad_flow()
    with torch.inference_mode():
        utils.plot_from_sample(
            utils.sample_flow(quad_flow, n_samples), "flow_sample_pretrain")

    print("INFO:\tTraining flow...")
    utils.train_flow(quad_flow, target_data, n_iter=10)
    with torch.inference_mode():
        utils.plot_from_sample(
            utils.sample_flow(quad_flow, n_samples), "flow_sample_posttrain")

    with torch.inference_mode():
        utils.benchmark_hep_style(quad_flow, target_data)

    print("INFO:\tFlow training complete. Please find plots under plots/")


if __name__ == "__main__":
    main()
