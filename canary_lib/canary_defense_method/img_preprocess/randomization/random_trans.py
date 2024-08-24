import numpy as np
import torch
from torchvision.transforms import ToTensor
from canary_lib.canary_defense_method.img_preprocess.randomization import randomization

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.trans_class(trans_name="random")
@sefi_component.config_params_handler(handler_target=ComponentType.TRANS, name="random",
                                      handler_type=ComponentConfigHandlerType.TRANS_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                      })
class Transform():
    def __init__(self, device='cuda'):
        self.device = device

    @sefi_component.trans(name="random", is_inclass=True)
    def img_transform(self, imgs):
        defense = randomization.Randomization(self.device)
        result = []
        for img in imgs:
            img = img.astype('float64')
            if not torch.is_tensor(img):
                img = torch.from_numpy(img)
            img = defense.forward(img.permute(2, 0, 1).unsqueeze(0))[0].permute(1, 2, 0)
            img = np.clip(img.numpy()/255, 0, 1, out=None)*255
            # img = img.cpu().detach().numpy().astype(np.uint8)
            # img = ToTensor()(img)
            result.append(img)
        return result

