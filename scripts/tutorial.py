from nflows import transforms, distributions, flows
import torch
import matplotlib.pyplot as plt

# Define an invertible transformation.
# Need to learn what these transforms do/why we might want to use these.
transform = transforms.CompositeTransform([
    transforms.MaskedAffineAutoregressiveTransform(features=2, hidden_features=4),
    transforms.RandomPermutation(features=2)
])

# Define a base distribution.
base_distribution = distributions.StandardNormal(shape=[2])
print(base_distribution)

# Combine into a flow.
flow = flows.Flow(transform=transform, distribution=base_distribution)
print(flow)

# Sample from the flow 
with torch.inference_mode():
    num_samples = 1000
    samples = flow.sample(num_samples)
    print(samples)
    print(type(samples))
    print(samples.shape)

    # Gotta plot them 
    plt.scatter(samples[:][0], samples[:][1])
    plt.savefig("dummy.png")

