import copy

import numpy as np
import torch

from canary_sefi.core.component.component_enum import ComponentConfigHandlerType, SubComponentType, \
    TransComponentAttributeType
from canary_sefi.core.component.default_component.model_getter import get_model
from canary_sefi.entity.dataset_info_entity import DatasetInfo, DatasetType
from canary_sefi.core.component.component_manager import SEFI_component_manager
from canary_sefi.core.component.default_component.params_handler import build_dict_with_json_args
from canary_sefi.core.function.basic.dataset.dataset_function import dataset_image_reader
from canary_sefi.evaluator.logger.adv_example_file_info_handler import find_adv_example_file_logs_by_attack_id, \
    find_adv_example_file_log_by_id
from canary_sefi.handler.image_handler.img_io_handler import save_pic_to_temp
from canary_sefi.handler.tools.cuda_memory_tools import check_cuda_memory_alloc_status
from canary_sefi.evaluator.logger.trans_file_info_handler import add_adv_trans_img_file_log


class Image_Transformer:
    def __init__(self, trans_name, trans_args, model_name=None, model_args=None, img_proc_args=None, run_device=None):
        self.trans_name = trans_name
        self.trans_args = trans_args
        self.trans_component = SEFI_component_manager.trans_method_list.get(trans_name)
        # 攻击处理参数JSON转DICT
        self.trans_args_dict = build_dict_with_json_args(self.trans_component,
                                                         ComponentConfigHandlerType.TRANS_CONFIG_PARAMS,
                                                         trans_args, run_device)
        self.trans_args_dict = self.trans_args_dict if self.trans_args_dict is not None else {}

        # 增加模型访问统计
        self.query_num = {
            "backward": 0,
            "forward": 0,
        }
        self.img_preprocessor = None
        # if model_name is not None:
        #     model_args = {}
        #     self.trans_args_dict['model'] = get_model(model_name, model_args, run_device, self)
        #
        #     model_component = SEFI_component_manager.model_list.get(model_name)
        #     # 图片处理参数JSON转DICT
        #     img_proc_args = {}
        #     self.img_proc_args_dict = build_dict_with_json_args(model_component,
        #                                                         ComponentConfigHandlerType.IMG_PROCESS_CONFIG_PARAMS,
        #                                                         img_proc_args, run_device)
        #     # 图片预处理
        #     self.img_preprocessor = model_component.get(SubComponentType.IMG_PREPROCESSOR, None, True)
        #     # 结果处理
        #     self.img_reverse_processor = model_component.get(SubComponentType.IMG_REVERSE_PROCESSOR, None, True)


        self.trans_func = self.trans_component.get(SubComponentType.TRANS_FUNC)

        # 判断转换方法的构造模式
        if self.trans_component.get(TransComponentAttributeType.IS_INCLASS) is True:
            # 构造类传入
            trans_class_builder = self.trans_component.get(SubComponentType.TRANS_CLASS)
            self.trans_class = trans_class_builder(**self.trans_args_dict)
            # 转换类初始化方法
            self.trans_init = self.trans_component.get(SubComponentType.TRANS_INIT, None)

    def adv_trans_4_img(self, img):
        if self.img_preprocessor is not None:  # 图片预处理器存在
            img = self.img_preprocessor(img, self.img_proc_args_dict)
        if self.trans_component.get(TransComponentAttributeType.IS_INCLASS) is True:
            img_trans = self.trans_func(self.trans_class, img)
        else:
            img_trans = self.trans_func(self.trans_args_dict, img)
        # 结果处理（一般是图片逆处理器）
        # if self.img_reverse_processor is not None:
        #     img_trans = self.img_reverse_processor(img_trans, {})
        return img_trans

    def destroy(self):
        del self.trans_class
        check_cuda_memory_alloc_status(empty_cache=True)


def adv_trans_4_img_batch(trans_name, trans_args, atk_log, model_name=None, model_args=None, img_proc_args=None, run_device=None, use_raw_nparray_data=False):
    trans_img_id_list = []
    # 查找攻击样本日志
    attack_id = atk_log['attack_id']
    all_adv_log = find_adv_example_file_logs_by_attack_id(attack_id)

    adv_img_cursor_list = []
    for adv_log in all_adv_log:
        adv_img_cursor_list.append(adv_log["adv_img_file_id"])

    # 读取攻击样本
    adv_dataset_info = DatasetInfo(None, None,
                                   dataset_type=DatasetType.ADVERSARIAL_EXAMPLE_RAW_DATA
                                   if use_raw_nparray_data else DatasetType.ADVERSARIAL_EXAMPLE_IMG,
                                   img_cursor_list=adv_img_cursor_list)

    # 构建图片转换器
    adv_trans = Image_Transformer(trans_name, trans_args, model_name.split("_")[0], model_args, img_proc_args, run_device)

    save_path = str(attack_id) + "/trans/" + trans_name + "/"

    def trans_iterator(imgs, img_log_ids, img_labels, save_raw_data=True):
        # 生成防御样本
        trans_results = adv_trans.adv_trans_4_img(copy.deepcopy(imgs))
        for index in range(len(trans_results)):
            img_log_id = img_log_ids[index]
            adv_img_file_log = find_adv_example_file_log_by_id(img_log_id)
            adv_img_file_id = adv_img_file_log['adv_img_file_id']
            if torch.is_tensor(trans_results[index]):
                trans_results[index] = trans_results[index].detach().cpu().numpy()
            trans_img_file_name = "adv_trans_{}.png".format(img_log_id)
            save_pic_to_temp(save_path, trans_img_file_name, trans_results[index], save_as_numpy_array=False)
            raw_file_name = None
            if save_raw_data:
                raw_file_name = "adv_trans_{}.npy".format(img_log_id)
                save_pic_to_temp(save_path, raw_file_name, trans_results[index], save_as_numpy_array=True)

            # 写入日志
            adv_trans_img_file_id = add_adv_trans_img_file_log(trans_name, attack_id, adv_img_file_id,
                                                               trans_img_file_name, raw_file_name)
            trans_img_id_list.append(adv_trans_img_file_id)

    dataset_image_reader(trans_iterator, adv_dataset_info)

    adv_trans.destroy()
    del adv_trans
    return trans_img_id_list
