# Define a custom exception for cyclic dependencies
class CyclicDependencyError(Exception):
    def __init__(self, loop_features):
        message = f"Cyclic dependency detected among the following features: {loop_features}. Feature dependencies must be acyclic."
        super().__init__(message)
        self.loop_features = loop_features
