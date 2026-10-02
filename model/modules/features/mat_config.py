import torch
from torch import nn

from .input_feature import InputFeature


class MatConfig(InputFeature):
    HASH = 0x3A5C91E7
    FEATURE_NAME = "MatConfig"
    INPUT_FEATURE_NAME = "MatConfig"
    MAX_ACTIVE_FEATURES = 4

    NUM_INPUTS = 2028
    NUM_REAL_FEATURES = 2028

    def __init__(self, num_outputs: int):
        super().__init__()

        self.num_outputs = num_outputs
        self.weight = nn.Parameter(
            torch.zeros(self.NUM_INPUTS, num_outputs, dtype=torch.float32)
        )

    def merged_weight(self) -> torch.Tensor:
        return self.weight

    @torch.no_grad()
    def coalesce(self) -> None:
        pass

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        pass

    @torch.no_grad()
    def init_weights(self) -> None:
        pass

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return self.weight.data.clone()

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        self.weight.data.copy_(export_weight)
