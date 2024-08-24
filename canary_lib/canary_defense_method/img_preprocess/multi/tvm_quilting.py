import torch

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType
from canary_lib.canary_defense_method.img_preprocess.quilting.quilting import quilting
from canary_lib.canary_defense_method.img_preprocess.tvm.tvm import reconstruct as tvm

sefi_component = SEFIComponent()


@sefi_component.trans_class(trans_name="tvm_quilting")
@sefi_component.config_params_handler(handler_target=ComponentType.TRANS, name="tvm_quilting",
                                      handler_type=ComponentConfigHandlerType.TRANS_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                      })
class Transform():
    def __init__(self):
        pass

    @sefi_component.trans(name="tvm_quilting", is_inclass=True)
    def img_transform(self, imgs):
        result = []
        for img in imgs:
            if not torch.is_tensor(img):
                img = torch.from_numpy(img)
            img = img / 255
            img = tvm(
                img, 0.05, 'none', 0.03
            ) * 255
            img = quilting(img, 2, 16)
            result.append(img)
        return result


