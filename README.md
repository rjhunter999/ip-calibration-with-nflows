# ip-calibration-with-nflows
Small project to investigate if the mismodelling in the LHCb simulation of the muon impact parameter resolution of Z decays can be corrected/described using normalising flows.

`notebooks/tutorial.ipynb` essentially implements the `nflows` [tutorial](https://pypi.org/project/nflows/) - I also looked in the `moons` tutorial for some help.
This was all neatened up into a script under `scripts/tutorial.py`.

`scripts/flow_to_lhcb_ip_sim.ipynb` shows my attempts at building a normalising flow that can flow from a Gaussian base distribution to the simulation log10(IP) distribution of Z-> mu mu decays in LHCb. I eventually converged on a quadratic spline flow which worked well.
