from dataclasses import dataclass
from typing import Dict, List, Optional

from torch import Tensor

@dataclass
class InteractDiffusionData:
    """
    Represents an entire ControlNet processing unit.
    """
    enabled: bool = True
    scheduled_sampling_rate: float = 1.0
    subject_phrases: List[str] = None
    object_phrases: List[str] = None
    action_phrases: List[str] = None
    subject_boxes: List[List[float]] = None
    object_boxes: List[List[float]] = None

    def __eq__(self, other):
        if not isinstance(other, InteractDiffusionData):
            return False

        return vars(self) == vars(other)