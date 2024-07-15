from nflows import transforms, distributions, flows
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import uproot

def make_1d_quad_flow():
    N_FEATURES = 1
    return flows.Flow(transform=transforms.PiecewiseRationalQuadraticCDF(shape=N_FEATURES, tails='linear', tail_bound=4, num_bins=3 ), distribution=distributions.StandardNormal(shape=[N_FEATURES]))


def plot_from_sample(samples, plot_sfx):
    _, ax = plt.subplots()
    hist, bins = np.histogram(samples, bins=100, range=(-4, 0))
    ax.bar(x=bins[:-1], height=hist, yerr=np.sqrt(hist), width=bins[1] - bins[0], label=plot_sfx)
    ax.set_xlabel('log10(IP / mm)')
    ax.set_ylabel('Candidates / bin')
    ax.legend()
    plt.savefig(f"plots/first_flow__{plot_sfx}.png")


def sample_flow(flow, n_samples):
    with torch.inference_mode():
        samples = flow.sample(n_samples)
        print(f"Taken {n_samples} samples from flow.\nsamples.shape: {samples.shape}")
        return samples


def train_flow(flow, target_data, n_iter, xrange=(-4, 0)):
    binning = dict(bins=100, range=xrange)
    
    # Setup the plot
    _, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax = ax.flatten() 
    i_fig = 0
    # Plot the target
    hist_target, bins = np.histogram(target_data, **binning)

    optimizer = torch.optim.Adam(flow.parameters())
    # Start training and adding to the plot periodically
    for i in range(n_iter+1):
        optimizer.zero_grad()
        loss = -flow.log_prob(inputs=target_data).mean()
        loss.backward()
        optimizer.step()
        ### Testing the thing
        with torch.inference_mode():
            if i % (n_iter/5) == 0:
                print(f"Loss at iteration {i}: {loss}")
                ax[i_fig].set_title(f"Iteration {i}: Loss = {loss:.5f}")
                ax[i_fig].bar(x=bins[:-1], height=hist_target, yerr=np.sqrt(hist_target), width=bins[1] - bins[0], label='MC', fill=False, edgecolor='blue')
                ax[i_fig].set_xlabel('log10(IP / mm)')
                ax[i_fig].set_ylabel('Candidates / bin')

                # Sample from the flow and plot the histogram
                samples = flow.sample(target_data.shape[0])
                pred_hist, bins = np.histogram(samples, **binning)
                ax[i_fig].bar(x=bins[:-1], height=pred_hist, yerr=np.sqrt(pred_hist), width=bins[1] - bins[0], label='Flow', fill=False, edgecolor='red')

                ax[i_fig].legend()
        
                i_fig+=1

    plt.savefig("plots/first_flow__training.png")

# Let's first load up the simulated data and put it into a 1D numpy array
def load_data(n_samples):
    file = uproot.open("data/tuple_for_training__Z.root")
    muon_prefix = "mup_" # could also use mum_
    branch = f'{muon_prefix}IP'
    selection = "(1>0)"
    sim_ip = file['DecayTree'].arrays(
        branch, cut=selection, library='np')[branch].astype(np.float64)

    sim_log10_ip = np.log10(sim_ip)[:, :n_samples]
    # take first N_SAMPLES from sim_log10_ip
    print(sim_log10_ip.shape)

    # Reshape was needed to swap the axes to match what the flow expects
    sim_log10_ip = torch.tensor(sim_log10_ip, dtype=torch.float32).reshape(-1, 1)
    print(sim_log10_ip.shape)
    return sim_log10_ip


def main():

    if not os.path.exists("plots"):
        os.makedirs("plots")

    print("INFO:\tLoading target data...")
    n_samples = 10000
    target_data = load_data(n_samples)
    plot_from_sample(target_data, "target_sample")

    torch.manual_seed(42)
    print("INFO:\tInstantiating spline flow...")
    quad_flow = make_1d_quad_flow()
    plot_from_sample(sample_flow(quad_flow, n_samples), "flow_sample_pretrain")

    print("INFO:\tTraining flow...")
    train_flow(quad_flow, target_data, n_iter=10000 )
    plot_from_sample(sample_flow(quad_flow, n_samples), "flow_sample_posttrain")
    print("INFO:\tFlow training complete. Please find plots under plots/")

if __name__ == "__main__":
    main()