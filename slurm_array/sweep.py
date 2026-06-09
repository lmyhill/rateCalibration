import json
import numpy as np
import pandas as pd
from scipy.stats import qmc

with open("config_sweep.json","r") as f:
    config = json.load(f)

application_domain = config["application_domain"]
input_metrics = config["inputMetrics"]

number_of_simulations = config["numberOfSimulations"]

sampled_metrics = {}
sampled_metrics.update(input_metrics)
sampled_metrics.update(application_domain)

sampler = qmc.LatinHypercube(d=len(sampled_metrics))
sample = sampler.random(n=number_of_simulations)

bounds = np.array([
    [
        metric_bounds["min_value"],
        metric_bounds["max_value"]
    ]
    for metric_bounds in sampled_metrics.values()
])

scaled_sample = qmc.scale(sample, bounds[:,0], bounds[:,1])

df = pd.DataFrame(
    scaled_sample,
    columns=sampled_metrics.keys()
)

df.to_csv("theta_samples_generated.csv",index=False)

print(df.head())
print(f"\nGenerated {len(df)} samples.")