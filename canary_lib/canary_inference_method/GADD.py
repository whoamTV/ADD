import copy

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentConfigHandlerType, ComponentType
from canary_lib.defense_attack import *

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="GADD", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(
    handler_target=ComponentType.ATTACK, name="GADD",
    handler_type=ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS,
    use_default_handler=True,
    params={
        "clip_min": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true", "def": "0.00"},
        "clip_max": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true", "def": "1.00"},
        "T": {"desc": "迭代攻击次数", "type": "INT", "required": "true", "def": "10"},
        "epsilon": {"desc": "对抗样本与原始输入图片的最大变化", "type": "FLOAT", "required": "true", "def": "0.06274"},
        "attack_type": {"desc": "攻击类型", "type": "SELECT", "selector": [{"value": "TARGETED", "name": "靶向"},{"value": "UNTARGETED", "name": "非靶向"}], "required": "true"},
        "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"}
    })
class MI_FGSM():
    def __init__(self, model, run_device, attack_type='UNTARGETED', clip_min=0, clip_max=1, T=10, epsilon=16/255, tlabel=None):
        self.model = model  # 待攻击的白盒模型
        self.T = T  # 迭代攻击轮数
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.clip_min = clip_min  # 像素值的下限
        self.clip_max = clip_max  # 像素值的上限
        self.attack_type = attack_type  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.device = run_device

    # 将图片进行clip
    def clip_value(self, x, ori_x):
        if self.epsilon is not None:
            x = torch.clamp((x - ori_x), -self.epsilon, self.epsilon) + ori_x
        x = torch.clamp(x, self.clip_min, self.clip_max)
        return x.data

    @sefi_component.attack(name="GADD", is_inclass=True, support_model=[], attack_type="WHITE_BOX")
    def attack(self, img, ori_labels, tlabels=None):
        batch_size = img.shape[0]
        tlabels = np.repeat(self.tlabel, batch_size) if tlabels is None else tlabels

        loss_ = torch.nn.CrossEntropyLoss()

        # 定义图片可获取梯度
        img.requires_grad = True
        # 克隆原始数据
        ori_img = img.clone()
        sum_grad = 0 # 累计梯度

        # 迭代攻击
        for iter in range(self.T):
            # 记录总输出，求和形式
            output = 0
            # 模型预测
            self.model.zero_grad()
            x = copy.deepcopy(img)

            # choose the defense to generate defense inference example
            x = jpeg_a(x)

            x.requires_grad = True
            img.requires_grad = True
            output = self.model(x)
            output_i = self.model(img)
            # 计算loss
            if self.attack_type == 'UNTARGETED':
                loss = loss_(output, torch.Tensor(ori_labels).to(self.device).long())  # 非靶向
                loss_i = -loss_(output_i, torch.Tensor(ori_labels).to(self.device).long())  # 非靶向
            else:
                loss = -loss_(output, torch.Tensor(tlabels).to(self.device).long())  # 靶向
                loss_i = loss_(output_i, torch.Tensor(ori_labels).to(self.device).long())

            # 反向传播
            loss_all = loss + loss_i
            loss_all.backward()
            grad = x.grad.data + img.grad.data
            # MIM
            grad = grad / torch.std(grad, dim=(1, 2), keepdim=True)
            grad = grad + sum_grad
            grad = grad / torch.std(grad, dim=(1, 2), keepdim=True)
            sum_grad = grad

            # 更新图像像素
            img.data = img.data + ((self.epsilon * 2) / self.T) * torch.sign(grad)
            img.data = self.clip_value(img, ori_img)



        return img