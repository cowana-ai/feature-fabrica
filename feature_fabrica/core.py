# core.py
import concurrent.futures
from collections import defaultdict

import numpy as np
from beartype import BeartypeConf, BeartypeStrategy, beartype
from easydict import EasyDict as edict
from graphviz import Digraph
from omegaconf import DictConfig

from feature_fabrica._internal.compute import (compile_all_transformations,
                                               compute_all_transformations)
from feature_fabrica.models import FeatureSpec, PromiseValue, THead, TNode
from feature_fabrica.promise_manager import get_promise_manager
from feature_fabrica.utils import get_logger, instantiate, verify_dependencies
from feature_fabrica.yaml_parser import load_yaml

logger = get_logger()
PROMISE_MANAGER = get_promise_manager()
# Dynamically create a new @slowmobeartype decorator enabling "full fat"
# O(n) type-checking.
# Type-check all items of the passed list. Do this only when you pretend
# to know in your guts that this list will *ALWAYS* be ignorably small.
slowmobeartype = beartype(conf=BeartypeConf(strategy=BeartypeStrategy.On))


class Feature:
    def __init__(self, name: str, spec: DictConfig, log_transformation_chain: bool):
        self.name = name

        self.spec = FeatureSpec(**spec)
        self.dependencies = self.spec.dependencies
        self.group = self.spec.group
        self.transformation = instantiate(self.spec.transformation)
        self.feature_value = PromiseValue(data_type=self.spec.data_type)

        self.log_transformation_chain = log_transformation_chain
        self.transformation_chain_head = THead()
        self.transformation_ptr = self.transformation_chain_head
        self.computed = False
        self.promised: bool = False


    def compile(self, dependencies: dict[str, "Feature"] | None = None) -> None:
        compile_all_transformations(self.transformation, dependencies)
        self.promised = PROMISE_MANAGER.is_promised_any(self.name)
        return

    @logger.catch(reraise=True)
    def compute(self, value: np.ndarray | None = None) -> np.ndarray:
        """Compute the feature value by applying its transformation.

        Parameters
        ----------
        value : Any, optional
            The input value to the transformation.

        Returns
        -------
        np.ndarray
            The computed feature value.
        """
        if self.promised and PROMISE_MANAGER.is_promised(self.name):
            PROMISE_MANAGER.pass_data(data=value, base_name=self.name)

        # Apply the transformation function if specified
        if self.transformation:
            try:
                result = compute_all_transformations(self.transformation, initial_value=value, get_intermediate_results=self.promised or self.log_transformation_chain)
                if self.promised or self.log_transformation_chain:
                    result, intermediate_results = result
                    for transformation_name, result_dict in intermediate_results:
                        if self.promised and PROMISE_MANAGER.is_promised(self.name, transformation_name):
                            PROMISE_MANAGER.pass_data(data=result_dict.value, base_name=self.name, suffix=transformation_name)

                        if self.log_transformation_chain:
                            self.update_transformation_chain(
                                transformation_name, result_dict
                            )

            except Exception as e:
                if self.log_transformation_chain:
                    transformation_chain_str = self.get_transformation_chain()
                    logger.debug(transformation_chain_str)
                raise e
            value = result.value

        self.feature_value(value)
        return self.feature_value.value  # type: ignore[attr-defined]

    @logger.catch(reraise=True)
    def __call__(self, value: np.ndarray | None = None) -> np.ndarray:
        result = self.compute(value)
        self.finalize_feature()
        return result

    def finalize_feature(self):
        self.computed = True
        PROMISE_MANAGER.delete_all_related_keys(self.name)
        self.promised = False

    def update_transformation_chain(self, transformation_name: str, result_dict: edict):
        """Update the transformation chain with the results of the latest transformation.

        Parameters
        ----------
        transformation_name : str
            The name of the transformation.
        result_dict : edict
            The result of the transformation.
        """
        assert isinstance(
            result_dict.value, np.ndarray
        ), f"result_dict.value has to be np.ndarray, {transformation_name} might have gone wrong!"
        start_time = result_dict.start_time
        end_time = result_dict.end_time

        transformation_node = TNode(
            transformation_name=transformation_name,
            start_time=start_time,
            end_time=end_time,
        )
        transformation_node.store_hash_and_shape(result_dict.value)
        transformation_node.finilize_metrics()
        self.transformation_ptr.next = transformation_node
        self.transformation_ptr = transformation_node

    def get_transformation_chain(self) -> str:
        assert self.log_transformation_chain, f"log_transformation_chain = {self.log_transformation_chain}, turn it to True to be able to see it!"
        current = self.transformation_chain_head.next
        chain_list = []
        while current:
            chain_list.append(
                f"(Transformation: {current.transformation_name}, Hash: {current.output_hash}, Shape: {current.shape}, Time taken: {current.time_taken} seconds)"
            )
            current = current.next
        return "Transformation Chain: " + " -> ".join(chain_list)


class FeatureManager:
    def __init__(
        self,
        config_path: str,
        config_name: str,
        parallel_execution: bool = True,
        log_transformation_chain: bool = True,
    ):
        self.feature_specs: DictConfig = load_yaml(
            config_path=config_path, config_name=config_name
        )
        self.parallel_execution = parallel_execution
        self.log_transformation_chain = log_transformation_chain

        self.independent_features: list[Feature] = []
        self.dependent_features: list[Feature] = []
        self.queue: dict[int, list[Feature]] = defaultdict(list)

        self.grouped_features: dict[str, list[Feature]] = defaultdict(list)

        self.features: edict = self._build_features()
        self.compile()

    @logger.catch(reraise=True)
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
            feature = Feature(
                name=name,
                spec=spec,
                log_transformation_chain=self.log_transformation_chain,
            )

            if not feature.dependencies:
                self.independent_features.append(feature)
            else:
                self.dependent_features.append(feature)

            if feature.group:
                self.grouped_features[feature.group].append(feature)


            features[name] = feature

        return features

    @logger.catch(reraise=True)
    def compile(self):
        """Identifies feature dependencies and the order in which Features are visited and computed.

        Returns
        -------
        None.
        """
        logger.info("Compiling features and feature dependencies...")

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

        self.compile_features()

    def compile_features(self):
        def compile_feature(feature: Feature):
            if not feature.dependencies:
                # Independent feature
                feature.compile(None)
            else:
                # Dependent feature
                dependencies = {
                    f_name: self.features[f_name] for f_name in feature.dependencies
                }
                feature.compile(dependencies=dependencies)

        for priority in sorted(self.queue.keys()):
            cur_features = self.queue[priority]
            if self.parallel_execution:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_feature = {
                        executor.submit(compile_feature, feature): feature
                        for feature in cur_features
                    }
                    for future in concurrent.futures.as_completed(future_to_feature):
                        future.result()
            else:
                for feature in cur_features:
                    compile_feature(feature=feature)

    def compute_single_feature(self, feature: Feature, value: np.ndarray | None = None):
        if value is not None:
            result = feature(value=value)
        else:
            result = feature()
        return feature.name, result

    @slowmobeartype
    def compute_features_with_validation(
        self, data_keys: list[str], data_values: list[np.ndarray], select_groups: list[str] | None = None
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

        for priority in sorted(self.queue.keys()):
            cur_features = self.queue[priority]

            if self.parallel_execution:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_feature = {
                        executor.submit(
                            self.compute_single_feature,
                            feature,
                            data[feature.name] if feature.name in data else None,
                        ): feature
                        for feature in cur_features
                    }
                    for future in concurrent.futures.as_completed(future_to_feature):
                        feature_name, result = future.result()
                        results[feature_name] = result
            else:
                for feature in cur_features:
                    feature_name, result = self.compute_single_feature(
                        feature=feature,
                        value=data[feature.name] if feature.name in data else None,
                    )
                    results[feature_name] = result

        if select_groups:
            results = {}
            for group in select_groups:
                features_group = self.grouped_features[group]
                for feature in features_group:
                    results[feature.name] = feature.feature_value.value

        return edict(results)

    def compute_features(self, data: dict[str, np.ndarray], select_groups: list[str] | None = None) -> edict:
        return self.compute_features_with_validation(
            list(data.keys()), list(data.values()), select_groups
        )

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
