import time

import numpy as np

from feature_fabrica.core import FeatureManager


def batch_data(data, batch_size):
    # Split data into batches of `batch_size`
    num_samples = len(next(iter(data.values())))
    for i in range(0, num_samples, batch_size):
        batch = {key: val[i:i + batch_size] for key, val in data.items()}
        yield batch

# Example data
data = {
    "feature_a": np.arange(1_000_000),
    "feature_b": np.arange(1_000_000),
}

# Initialize FeatureManager
feature_manager = FeatureManager(
    config_path="./examples", config_name="math_features", log_transformation_chain=False
)

# Define batch size
batch_size = 512

# Store results
all_results = []
batch_times = []

# Process each batch
for batch in batch_data(data, batch_size):
    start_time = time.time()
    results = feature_manager.compute_features(batch)
    end_time = time.time()

    all_results.append(results)
    batch_times.append(end_time - start_time)

# Optionally: Combine results across batches (this depends on your output format)
# For example, if `results` are dictionaries, merge them:
final_results = {key: np.concatenate([result[key] for result in all_results], axis=0) for key in all_results[0]}

# Compute total time spent
total_time = sum(batch_times)
print(f"Total time spent on batched computation: {total_time} seconds")
