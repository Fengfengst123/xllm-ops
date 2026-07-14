import torch
import torch_npu
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


def _inputs(input_data: InputDataset):
    return (
        input_data.kwargs["x1"],
        input_data.kwargs["x2"],
        input_data.kwargs["gamma"],
        float(input_data.kwargs["epsilon"]),
    )


@register("bare_add_rms_norm_perf")
class BareAddRmsNormPerfApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x1, x2, gamma, epsilon = _inputs(input_data)
        outputs = torch_npu.npu_add_rms_norm(x1, x2, gamma, epsilon)
        return outputs if with_output else None


@register("add_plus_add_rms_norm_perf")
class AddPlusAddRmsNormPerfApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x1, x2, gamma, epsilon = _inputs(input_data)
        gamma_plus_one = gamma + 1.0
        outputs = torch_npu.npu_add_rms_norm(x1, x2, gamma_plus_one, epsilon)
        return outputs if with_output else None
