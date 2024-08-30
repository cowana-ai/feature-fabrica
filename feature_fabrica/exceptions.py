# Define a custom exception for cyclic dependencies
class CyclicDependencyError(Exception):
    def __init__(self, loop_features):
        message = f"Cyclic dependency detected among the following features: {loop_features}. Feature dependencies must be acyclic."
        super().__init__(message)
        self.loop_features = loop_features


class FeatureNotComputedError(Exception):
    def __init__(self, f_name: str):
        message = f"The feature {f_name} must be computed before finalizing metrics."
        super().__init__(message)
        self.f_name = f_name
