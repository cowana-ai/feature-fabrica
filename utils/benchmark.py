import argparse
import time

import numpy as np

from feature_fabrica.core import FeatureManager


def batch_data(data, batch_size):
    """Split data into batches of `batch_size`."""
    num_samples = len(next(iter(data.values())))
    for i in range(0, num_samples, batch_size):
        batch = {key: val[i:i + batch_size] for key, val in data.items()}
        yield batch

def main(config_name, batch_size, total_samples):
    # Generate example data based on the total number of samples
    data = {
        "feature_a": np.arange(total_samples),
        "feature_b": np.arange(total_samples),
    }

    # Initialize FeatureManager
    feature_manager = FeatureManager(
        config_path="./examples", config_name=config_name, log_transformation_chain=False
    )

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
    final_results = {key: np.concatenate([result[key] for result in all_results], axis=0) for key in all_results[0]}

    # Compute total time spent
    total_time = sum(batch_times)
    print(f"Total time spent on batched computation: {total_time:.4f} seconds")
    return final_results

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Batched feature computation.")
    parser.add_argument("--config_name", type=str, required=True, help="Name of the feature configuration.")
    parser.add_argument("--batch_size", type=int, default=512, help="Size of each batch.")
    parser.add_argument("--total_samples", type=int, default=1_000_000, help="Total number of samples to generate.")

    # Parse arguments
    args = parser.parse_args()

    # Call main function with parsed arguments
    final_results = main(config_name=args.config_name, batch_size=args.batch_size, total_samples=args.total_samples)
