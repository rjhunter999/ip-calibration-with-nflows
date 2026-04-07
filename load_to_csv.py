import uproot

sample_type = "Z"
file = uproot.open(f"data/tuple_for_training__{sample_type}.root")
muon_prefix = "mup_"  # could also use mum_
branch = f"{muon_prefix}IP"
ip_df = file["DecayTree"].arrays(branch, library="pandas", entry_stop=1000)
# take first n_samples from sim_log10_ip
# log10_ip_arr = np.log10(ip_arr)[:, :n_samples]

# now save as csv via a dataframe
print(ip_df)

ip_df.to_csv(f"data/tuple_for_training__{sample_type}.csv")
