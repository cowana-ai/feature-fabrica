# models.py
import hashlib
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel


class TNode(BaseModel):
    """Transformation node for tracking transformation metadata."""
    transformation_name: str
    start_time: float
    end_time: float
    shape: tuple | None = None
    time_taken: float | None = None
    output_hash: str | None = None
    next: Optional["TNode"] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the node and its next nodes to a dictionary."""
        node_dict = {
            "transformation_name": self.transformation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "shape": self.shape,
            "time_taken": self.time_taken,
            "output_hash": self.output_hash,
        }
        if self.next:
            node_dict["next"] = self.next.to_dict()
        return node_dict

    def compute_hash(self, data: np.ndarray) -> str:
        """Compute a hash for the output data of the transformation."""
        data_bytes = data.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()

    def store_hash_and_shape(self, output_data: np.ndarray):
        """Store the hash and shape of the output data."""
        self.shape = output_data.shape
        self.output_hash = self.compute_hash(output_data)

    def finalize_metrics(self):
        """Finalize transformation timing metrics."""
        self.time_taken = self.end_time - self.start_time



class THead(BaseModel):
    next: TNode | None = None
