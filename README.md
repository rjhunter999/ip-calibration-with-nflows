# ip-calibration-with-nflows
Small project to investigate if the mismodelling in the LHCb simulation of the muon impact parameter resolution of Z decays can be corrected/described using normalising flows.

Studies here have shown that this can be done, for example the below plot shows a histogram of the LHCb data, and then histograms of the corrected and uncorrected simulation, where the correction is a composite transformation derived from a trained flow. The ratio of the data to the corrected simulation is shown in the lower panel, giving generally good agreement within statistical uncertainty of the samples, although struggling a little with the shape of high-IP tail.

![Example of IP correction](./readme_plots/example.png)

The repository is divided into exploratory Jupyter notebooks under `notebooks/` and python scripts under `scripts/`. The latter are culminations of the former that could be used in production.

`notebooks/tutorial.ipynb` essentially implements the `nflows` [tutorial](https://pypi.org/project/nflows/) - I also looked in the `moons` tutorial for some help.
This was all neatened up into a script under `scripts/tutorial.py`.

`notebooks/flow_to_lhcb_ip_sim.ipynb` shows my attempts at building a normalising flow that can flow from a Gaussian base distribution to the simulation log10(IP) distribution of Z-> mu mu decays in LHCb. I eventually converged on a quadratic spline flow which worked well.

`notebooks/extracting_transformations.ipynb` shows investigations in how to extract the trained transformation from a flow, its inverse, and how to then put two transformations together.

Training a flow to derive the correction to the simulation and get the above plot is achieved by `scripts/ip_correction_flow.py`. Note that this script has a `--test` argument that runs on smaller samples and trains for less iterations for quick prototyping.

## Setup instructions

Before running any of the python scripts, setup the environment with:

```bash
source setup_python_env.sh
```

Then you can execute any of the python scripts within from the top-level directory e.g.

```
[user@users-computer ip-calibration-with-nflows]$ ./scripts/tutorial.py
```

The repository utilises `pre-commit` to handle formatting of code. Please follow the installation [instructions](https://pre-commit.com/) if you don't already have `pre-commit` to install it in your checkout of the repository.
