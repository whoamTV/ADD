import math
import numpy as np
import torch
from torchvision.transforms import ToTensor

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
from PIL import Image

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType
sefi_component = SEFIComponent()


@sefi_component.trans_class(trans_name="quantize_jpeg")
@sefi_component.config_params_handler(handler_target=ComponentType.TRANS, name="quantize_jpeg",
                                      handler_type=ComponentConfigHandlerType.TRANS_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                      })
class Transform():
    def __init__(self):
        pass

    @sefi_component.trans(name="quantize_jpeg", is_inclass=True)
    def img_transform(self, imgs):
        result = []
        for img in imgs:
            if not torch.is_tensor(img):
                img = torch.from_numpy(img)
            img = self._quantize_img(img, 3)
            img = self._jpeg_compression(img)
            result.append(img)
        return result

    @staticmethod
    def _jpeg_compression(im, quality=50):
        assert torch.is_tensor(im)
        im = im.cpu().detach().numpy()
        im = im.astype(np.uint8)
        im = Image.fromarray(im)
        savepath = BytesIO()
        im.save(savepath, "JPEG", quality=quality)
        im = Image.open(savepath)
        im = ToTensor()(im)
        im = im.permute(1, 2, 0)
        im *= 255
        return im

    @staticmethod
    def _quantize_img(im, depth=3):
        assert torch.is_tensor(im)
        im /= 255
        N = int(math.pow(2, depth))
        im = (im * N).round()
        im = im / N
        im *= 255
        return im
