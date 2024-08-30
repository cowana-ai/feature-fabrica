# core.py
from .models import FeatureSpec, FeatureValue, TNode, THead
from typing import Any
from easydict import EasyDict as edict
from .yaml_parser import load_yaml
from collections import defaultdict
from omegaconf import OmegaConf
from hydra.utils import instantiate
from graphviz import Digraph
from .utils import get_logger, verify_dependencies

logger = get_logger()


class Feature:
    def __init__(self, name: str, spec: OmegaConf):
        self.name = name
        self.feature_value = None
        self.spec = FeatureSpec(**spec)
        self.dependencies = self.spec.dependencies
        self.transformation = instantiate(self.spec.transformation)
        self.transformation_chain = THead()
        self.transformation_ptr = self.transformation_chain

        self.computed = False

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
                raise
            value = result_dict.value

        else:
            assert dependencies is None, "Derived features must have transformations!"

        # Validate the final result with FeatureValue
        self.feature_value = FeatureValue(value=value, data_type=self.spec.data_type)  # type: ignore[assignment]
        self.computed = True
        return self.feature_value.value  # type: ignore[attr-defined]

    def update_transformation_chain(
        self, transformation_name: str, result_dict: dict[str, Any]
    ):
        transformation_node = TNode(
            transformation_name=transformation_name, **result_dict
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
        self.feature_specs: OmegaConf = load_yaml(
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

        features = {}
        for name, spec in self.feature_specs.items():
            feature = Feature(name, spec)

            if not feature.dependencies:
                self.independent_features.append(feature)
            else:
                self.dependent_features.append(feature)

            features[name] = feature

        return edict(features)

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

    def compute_features(self, data: dict[str, Any]) -> edict:
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
        results = {}

        # A single pass over the queue to reduce recomputation
        for priority in sorted(self.queue.keys()):
            cur_features = self.queue[priority]

            for feature in cur_features:
                if not feature.dependencies:
                    # Independent feature
                    result = feature.compute(value=data[feature.name])
                else:
                    # Dependent feature
                    dependencies = {
                        f_name: self.features[f_name] for f_name in feature.dependencies
                    }
                    result = feature.compute(dependencies=dependencies)

                results[feature.name] = result

        return edict(results)
