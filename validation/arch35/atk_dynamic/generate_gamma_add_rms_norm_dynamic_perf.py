from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("gamma_add_rms_norm_dynamic_perf")
class GammaAddRmsNormDynamicPerfGenerator(CaseGenerator):
    """Keep ATK-generated x1 shapes and enforce only operator dependencies."""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        x1, x2, gamma, epsilon, add_gamma_offset, kernel_type = case_config.inputs

        x2.dtype = x1.dtype
        x2.shape = list(x1.shape)
        x2.range_values = list(x1.range_values)

        gamma.dtype = x1.dtype
        gamma.shape = [x1.shape[-1]]
        gamma.range_values = [-0.5, 0.5]

        epsilon.range_values = [1.0e-6]
        add_gamma_offset.range_values = [True]
        kernel_type.range_values = [2]

        case_config.name = "GammaAddRmsNorm"
        case_config.aclnn_name = "GammaAddRmsNorm"
        case_config.api_type = "aclnn_gamma_add_rms_norm"
        case_config.aclnn_api_type = "sample_aclnn_api"
        case_config.standard.perf = "not_key"
        return case_config
