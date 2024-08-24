import torch
import numpy as np
import eagerpy as ep
from foolbox import PyTorchModel
from foolbox.attacks import LinearSearchBlendedUniformNoiseAttack
from foolbox.criteria import TargetedMisclassification, Misclassification
from canary_lib.canary_inference_method.DADD_core import DADD_Detection
from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType
from canary_lib.defense_attack import *

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="DADD", perturbation_budget_var_name=None)
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="DADD",
                                      handler_type=ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={})
class BoundaryAttack:
    def __init__(self, model, run_device, attack_type='UNTARGETED', tlabel=1, clip_min=0, clip_max=1, epsilon=None,
                 max_iterations=25000,
                 spherical_step=0.01,
                 source_step=0.01,
                 source_step_convergence=1e-07,
                 step_adaptation=1.5,
                 update_stats_every_k=10):
        self.model = PyTorchModel(model, bounds=(clip_min, clip_max), device=run_device)
        self.device = run_device if run_device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.attack_type = attack_type
        self.tlabel = tlabel
        self.init_attack = LinearSearchBlendedUniformNoiseAttack(steps=50)
        self.steps = max_iterations
        self.spherical_step = spherical_step
        self.source_step = source_step
        self.source_step_convergence = source_step_convergence
        self.step_adaptation = step_adaptation
        self.update_stats_every_k = update_stats_every_k

    @sefi_component.attack(name="DADD", is_inclass=True, support_model=[])
    def attack(self, imgs, ori_labels, tlabels=None):
        batch_size = imgs.shape[0]
        tlabels = np.repeat(self.tlabel, batch_size) if tlabels is None else tlabels

        # 转为PyTorch变量
        tlabels = ep.astensor(torch.from_numpy(np.array(tlabels)).to(self.device))
        ori_labels = ep.astensor(torch.from_numpy(np.array(ori_labels)).to(self.device))
        imgs = ep.astensor(imgs)

        # 实例化攻击类
        attack = DADD_Detection(init_attack=self.init_attack, steps=self.steps,
                                spherical_step=self.spherical_step,
                                source_step=self.source_step,
                                source_step_convergence=self.source_step_convergence,
                                step_adaptation=self.step_adaptation,
                                tensorboard=False,
                                update_stats_every_k=self.update_stats_every_k,
                                # choose the defense to generate defense inference example
                                defense=jpeg_a)
        if self.attack_type == 'UNTARGETED':
            criterion = Misclassification(labels=ori_labels)
            raw, clipped, is_adv = attack(self.model, imgs, criterion, epsilons=None)
        else:
            criterion = TargetedMisclassification(target_classes=tlabels)
            raw, clipped, is_adv = attack(self.model, imgs, criterion, epsilons=None)

        # 由EagerPy张量转化为PyTorch Native张量
        adv_img = raw.raw
        return adv_img
