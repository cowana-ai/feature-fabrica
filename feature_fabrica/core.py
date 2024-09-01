# core.py
import concurrent.futures
from collections import defaultdict
from typing import Any

import numpy as np
from beartype import BeartypeConf, BeartypeStrategy, beartype
from easydict import EasyDict as edict
from graphviz import Digraph
from hydra.utils import instantiate
from omegaconf import DictConfig

from feature_fabrica.exceptions import FeatureNotComputedError
from feature_fabrica.models import FeatureSpec, FeatureValue, THead, TNode
from feature_fabrica.utils import get_logger, verify_dependencies
from feature_fabrica.yaml_parser import load_yaml

logger = get_logger()
# Dynamically create a new @slowmobeartype decorator enabling "full fat"
# O(n) type-checking.
# Type-check all items of the passed list. Do this only when you pretend
# to know in your guts that this list will *ALWAYS* be ignorably small.
slowmobeartype = beartype(conf=BeartypeConf(strategy=BeartypeStrategy.On))


class Feature:
    def __init__(self, name: str, spec: DictConfig):
        self.name = name
        self.feature_value = None
        self.spec = FeatureSpec(**spec)
        self.dependencies = self.spec.dependencies
        self.transformation = instantiate(self.spec.transformation)
        self.transformation_chain = THead()
        self.transformation_ptr = self.transformation_chain

        self.computed = False

    @logger.catch
    def compute(self, value: Any = 0, dependencies: dict[str, "Feature"] | None = None):
        # Apply the transformation function if specified
        if self.transformation:
            try:
                prev_value = value
                for (
                    transformation_name,
                    transformation_obj,
                ) in self.transformation.items():
                    expects_data = transformation_obj.compile(dependencies)
                    if expects_data:
                        result_dict = transformation_obj(prev_value)
                    else:
                        result_dict = transformation_obj()
                    prev_value = result_dict.value

                    self.update_transformation_chain(transformation_name, result_dict)

            except Exception as e:
                transformation_chain_str = self.get_transformation_chain()
                logger.debug(transformation_chain_str)
                logger.error(
                    f"An error occurred during the transformation {transformation_name}: {e}"
                )
                raise e
            value = result_dict.value

        else:
            assert dependencies is None, "Derived features must have transformations!"

        self.feature_value = FeatureValue(value=value, data_type=self.spec.data_type)  # type: ignore[assignment]
        self.computed = True
        return self.feature_value.value  # type: ignore[attr-defined]

    def update_transformation_chain(self, transformation_name: str, result_dict: edict):
        if len(result_dict.value.shape) == 1:
            value = np.random.choice(result_dict.value, 1)
        else:
            value = None
        start_time = result_dict.start_time
        end_time = result_dict.end_time

        transformation_node = TNode(
            transformation_name=transformation_name,
            value=value,
            start_time=start_time,
            end_time=end_time,
        )
        transformation_node.finilize_metrics()
        self.transformation_ptr.next = transformation_node
        self.transformation_ptr = transformation_node

    def get_transformation_chain(self) -> str:
        current = self.transformation_chain.next
        chain_list = []
        while current:
            chain_list.append(
                f"(Transformation: {current.transformation_name}, Value: {current.value},  Time taken: {current.time_taken} seconds)"
            )
            current = current.next
        return "Transformation Chain: " + " -> ".join(chain_list)


class FeatureManager:
    def __init__(self, config_path: str, config_name: str):
        self.feature_specs: DictConfig = load_yaml(
            config_path=config_path, config_name=config_name
        )

        self.independent_features: list[Feature] = []
        self.dependent_features: list[Feature] = []
        self.queue: dict[int, list[Feature]] = defaultdict(list)

        self.features: edict = self._build_features()
        self.compile()

    @logger.catch
    def _build_features(self) -> edict:
        """Builds features. Separates features into dependent_features and independent_features features.

        Returns
        -------
        edict
            Dictionary:
                key - > feature name (string)
                value -> feature (Feature class).
        """
        logger.info("Building features from feature definition YAML")

        features = edict()
        for name, spec in self.feature_specs.items():
            feature = Feature(name, spec)

            if not feature.dependencies:
                self.independent_features.append(feature)
            else:
                self.dependent_features.append(feature)

            features[name] = feature

        return features

    @logger.catch
    def compile(self):
        """Identifies feature dependencies and the order in which Features are visited and computed.

        Returns
        -------
        None.
        """
        logger.info("Compiling feature dependencies")

        dependencies_count = defaultdict(int)
        visited = defaultdict(int)
        # Initialize independent features
        for feature in self.independent_features:
            dependencies_count[feature.name] = 1
            visited[feature.name] = 1

        # Resolve dependent features
        for feature in self.dependent_features:
            if dependencies_count[feature.name] != 0:
                continue

            cur_feature_depends = [
                (f_name, dependencies_count[f_name]) for f_name in feature.dependencies
            ]
            if 0 not in [x[1] for x in cur_feature_depends]:
                dependencies_count[feature.name] = sum(
                    [x[1] for x in cur_feature_depends]
                )
            else:
                # Handle unresolved dependencies using a stack
                stack = [
                    f_name
                    for f_name in feature.dependencies
                    if dependencies_count[f_name] == 0
                ]
                while stack:
                    f_node_name = stack.pop()
                    if visited[f_node_name]:
                        continue

                    # Mark this node as visited
                    visited[f_node_name] = 1

                    # Get the feature object by its name
                    f_node = self.features[f_node_name]

                    # Resolve dependencies of this node
                    node_feature_depends = [
                        (f_name, dependencies_count[f_name])
                        for f_name in f_node.dependencies
                    ]

                    if 0 in [x[1] for x in node_feature_depends]:
                        # If there are still unresolved dependencies, push back on stack
                        stack.append(f_node_name)
                        for dep_name, dep_count in node_feature_depends:
                            if dep_count == 0 and not visited[dep_name]:
                                stack.append(dep_name)
                    else:
                        # All dependencies resolved, update count
                        dependencies_count[f_node_name] = sum(
                            [x[1] for x in node_feature_depends]
                        )

                # Finally, update the current feature's dependency count
                dependencies_count[feature.name] = sum(
                    [dependencies_count[f_name] for f_name in feature.dependencies]
                )

        verify_dependencies(dependencies_count)
        for f_name, level in dependencies_count.items():
            self.queue[level].append(self.features[f_name])

    def get_visual_dependency_graph(
        self, save_plot: bool = False, output_file: str = "feature_dependencies"
    ):
        dot = Digraph(comment="Feature Dependencies")

        # Add nodes
        for feature in self.features.values():
            dot.node(feature.name)

        # Add edges
        for feature in self.features.values():
            for dependency in feature.dependencies:
                dot.edge(dependency, feature.name)

        if save_plot:
            # Save and render the graph
            dot.render(output_file, format="png")
            logger.info(f"Dependencies graph saved as {output_file}.png")
        return dot

    def compute_single_feature(self, feature: Feature, data: dict[str, np.ndarray]):
        # Check if the feature has dependencies
        if not feature.dependencies:
            # Independent feature
            result = feature.compute(value=data[feature.name])
        else:
            # Dependent feature
            dependencies = {
                f_name: self.features[f_name] for f_name in feature.dependencies
            }
            result = feature.compute(dependencies=dependencies)
        return feature.name, result

    @slowmobeartype
    def compute_features_beartype(
        self, data_keys: list[str], data_values: list[np.ndarray]
    ) -> edict:
        """

        Parameters
        ----------
        data : dict[str, Any]
            Data point.

        Returns
        -------
        Dictionary
            Processed data point with derived features as well.

        """
        data = dict(zip(data_keys, data_values))
        results = {}

        # A single pass over the queue to reduce recomputation
        for priority in sorted(self.queue.keys()):
            cur_features = self.queue[priority]

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_feature = {
                    executor.submit(self.compute_single_feature, feature, data): feature
                    for feature in cur_features
                }
                for future in concurrent.futures.as_completed(future_to_feature):
                    feature_name, result = future.result()
                    results[feature_name] = result

        return edict(results)

    def compute_features(self, data: dict[str, np.ndarray]) -> edict:
        return self.compute_features_beartype(list(data.keys()), list(data.values()))

    def report(self):
        logger.info("Generating feature transformation report...")

        bottlenecks = []

        for feature in self.features.values():
            f_name = feature.name

            if not feature.computed:
                logger.error(
                    f"Feature '{f_name}' isn't computed! Unable to generate a report for it."
                )
                raise FeatureNotComputedError(f_name)

            total_transformations = len(feature.transformation)
            # Calculate the total compute time for the feature
            total_time = 0.0
            current = feature.transformation_chain.next
            while current:
                total_time += current.time_taken
                current = current.next

            # Log the report for each transformation
            logger.info(f"Feature '{f_name}':")
            current = feature.transformation_chain.next
            chain_list = []
            while current:
                transformation_name = current.transformation_name
                time_taken = current.time_taken
                percentage = (time_taken / total_time) * 100 if total_time > 0 else 0
                chain_list.append((transformation_name, time_taken, percentage))

                # Log and detect bottlenecks
                # TODO: configurable?
                if total_transformations >= 3:
                    if percentage > 80:
                        logger.error(
                            f"  - {transformation_name}: {time_taken:.4f} seconds ({percentage:.2f}%) [SEVERE BOTTLENECK]"
                        )
                        bottlenecks.append((f_name, transformation_name, "SEVERE"))
                    elif percentage > 50:
                        logger.warning(
                            f"  - {transformation_name}: {time_taken:.4f} seconds ({percentage:.2f}%) [MODERATE BOTTLENECK]"
                        )
                        bottlenecks.append((f_name, transformation_name, "MODERATE"))
                    else:
                        logger.info(
                            f"  - {transformation_name}: {time_taken:.4f} seconds ({percentage:.2f}%)"
                        )
                else:
                    logger.info(
                        f"  - {transformation_name}: {time_taken:.4f} seconds ({percentage:.2f}%) [NO BOTTLENECK]"
                    )

                current = current.next

        # Summary of Bottlenecks
        if bottlenecks:
            logger.info("Bottleneck Summary:")
            for feature_name, transformation_name, severity in bottlenecks:
                logger.info(
                    f"Feature '{feature_name}' has a {severity} bottleneck in '{transformation_name}'."
                )

        logger.info("Feature transformation report generated successfully.")
