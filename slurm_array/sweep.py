import json
import numpy as np
import pandas as pd
from scipy.stats import qmc


def extract_sampling_parameters(config):
    """
    Recursively find all parameters marked for sampling.

    Returns:
        dict:
            {
                parameter_name: {
                    min_value: ...,
                    max_value: ...
                }
            }
    """

    sampled_parameters = {}

    def recurse(settings):

        for key, value in settings.items():

            if isinstance(value, dict):

                if value.get("sample", False):

                    sampled_parameters[key] = {
                        "min_value": value["min_value"],
                        "max_value": value["max_value"]
                    }

                else:
                    recurse(value)

    recurse(config)

    return sampled_parameters


# ----------------------------------------------------------
# Read configuration
# ----------------------------------------------------------

with open("config_sweep.json", "r") as f:
    config = json.load(f)


number_of_simulations = config["numberOfSimulations"]


# ----------------------------------------------------------
# Find sampled parameters
# ----------------------------------------------------------

sampled_metrics = extract_sampling_parameters(config)


if len(sampled_metrics) == 0:
    raise RuntimeError(
        "No sampled parameters found. "
        "Check that parameters have \"sample\": true."
    )


print("Sampling parameters:")
for name, bounds in sampled_metrics.items():
    print(
        f"  {name}: "
        f"{bounds['min_value']} -> {bounds['max_value']}"
    )


# ----------------------------------------------------------
# Generate Latin hypercube samples
# ----------------------------------------------------------

sampler = qmc.LatinHypercube(
    d=len(sampled_metrics)
)

sample = sampler.random(
    n=number_of_simulations
)


bounds = np.array(
    [
        [
            metric_bounds["min_value"],
            metric_bounds["max_value"]
        ]
        for metric_bounds in sampled_metrics.values()
    ]
)


scaled_sample = qmc.scale(
    sample,
    bounds[:, 0],
    bounds[:, 1]
)


df = pd.DataFrame(
    scaled_sample,
    columns=sampled_metrics.keys()
)


# ----------------------------------------------------------
# Save samples
# ----------------------------------------------------------

output_file = config["samplingSettings"]["inputFile"]

df.to_csv(
    output_file,
    index=False
)


print(df.head())
print(
    f"\nGenerated {len(df)} samples."
)
print(
    f"Saved to {output_file}"
)