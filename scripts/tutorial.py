from nflows import transforms, distributions, flows
import torch
import matplotlib.pyplot as plt
import os

def make_basic_flow(hidden_features=4):
    # Need to learn what these transforms do/why we might want to use these.
    transform = transforms.CompositeTransform([
        transforms.MaskedAffineAutoregressiveTransform(features=2, hidden_features=hidden_features),
        transforms.RandomPermutation(features=2)
    ])
    base_distribution = distributions.StandardNormal(shape=[2])
    flow = flows.Flow(transform=transform, distribution=base_distribution)
    return flow

def plot_from_sample(samples, plot_sfx):
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.savefig(f"plots/tutorial__{plot_sfx}.png")


def sample_flow(flow, n_samples):
    with torch.inference_mode():
        samples = flow.sample(n_samples)
        print(f"Taken {n_samples} samples from flow.\nsamples.shape: {samples.shape}")
        return samples

def train_flow(flow, target):
    optimizer = torch.optim.Adam(flow.parameters())
    n_iter = 5000
    for i in range(n_iter):
        #flow.train() # Put it in training mode.

        # TODO Need to understand why we zero_grad() before calculating the loss.
        # TODO also need to understand the loss
        # TODO why no forward pass?
        optimizer.zero_grad()
        loss = -flow.log_prob(inputs=target).mean()
        loss.backward()
        optimizer.step()

        ### Testing the thing
        # flow.eval() # Put it in evaluation mode.
        with torch.inference_mode():
            if i % 1000 == 0:
                print(f"Loss at iteration {i}: {loss}")
                xline = torch.linspace(-10, 10, 100)
                yline = torch.linspace(-10, 10, 100)
                xgrid, ygrid = torch.meshgrid(xline, yline)
                xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

                zgrid = flow.log_prob(xyinput).exp().reshape(100, 100)

                plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
                plt.title('iteration {}'.format(i + 1))
                plt.savefig(f"plots/tutorial__flow_training__iteration{i}.png")

def main():
    if not os.path.exists("plots"):
        os.makedirs("plots")

    n_samples = 1000

    # Generate some target data.
    # Shift and scale it to make it different from the base distribution.
    print("INFO:\tGenerating target data...")
    torch.manual_seed(42)
    target = 2.0 * torch.randn(n_samples, 2) + 5
    plot_from_sample(target, "target_sample")

    print("INFO:\tInstantiating flow...")
    flow = make_basic_flow()
    plot_from_sample(sample_flow(flow, n_samples), "flow_sample_pretrain")

    print("INFO:\tTraining flow...")
    train_flow(flow, target)
    plot_from_sample(sample_flow(flow, n_samples), "flow_sample_posttrain")
    print("INFO:\tFlow training complete. Please find plots under plots/")

if __name__ == "__main__":
    main()