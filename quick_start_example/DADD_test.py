from canary_sefi.core.function.enum.multi_db_mode_enum import MultiDatabaseMode
from canary_sefi.core.function.helper.multi_db import use_multi_database
from canary_sefi.service.security_evaluation import SecurityEvaluation
from canary_sefi.task_manager import task_manager
from canary_sefi.core.component.component_manager import SEFI_component_manager
from canary_lib import canary_lib  # Canary Lib

SEFI_component_manager.add_all(canary_lib)
# SwinTransformer ConvNext DenseNet ResNeXt
batch_size_m = 50
if __name__ == "__main__":
    example_config = {
        "dataset_size": 2, "dataset": "ILSVRC-2012",
        "dataset_seed": 40376958655838027,
        "attacker_list": {
            "DADD": [
                "Alexnet(ImageNet)",
                "SwinTransformer(ImageNet)",
                "ConvNext(ImageNet)",
                "DenseNet(ImageNet)",
                "ResNeXt(ImageNet)"
            ],
        },
        "model_list": [
            "Alexnet(ImageNet)",
            "SwinTransformer(ImageNet)",
            "ConvNext(ImageNet)",
            "DenseNet(ImageNet)",
            "ResNeXt(ImageNet)"
        ],
        "trans_list": {
            "DADD": {
                "jpeg": [
                    "Alexnet(ImageNet)",
                    "SwinTransformer(ImageNet)",
                    "ConvNext(ImageNet)",
                    "DenseNet(ImageNet)",
                    "ResNeXt(ImageNet)"
                ],
                "tvm": [
                    "Alexnet(ImageNet)",
                    "SwinTransformer(ImageNet)",
                    "ConvNext(ImageNet)",
                    "DenseNet(ImageNet)",
                    "ResNeXt(ImageNet)"
                ],
                "quantize": [
                    "Alexnet(ImageNet)",
                    "SwinTransformer(ImageNet)",
                    "ConvNext(ImageNet)",
                    "DenseNet(ImageNet)",
                    "ResNeXt(ImageNet)"
                ],
                "random": [
                    "Alexnet(ImageNet)",
                    "SwinTransformer(ImageNet)",
                    "ConvNext(ImageNet)",
                    "DenseNet(ImageNet)",
                    "ResNeXt(ImageNet)"
                ],
                "quilting": [
                    "Alexnet(ImageNet)",
                    "SwinTransformer(ImageNet)",
                    "ConvNext(ImageNet)",
                    "DenseNet(ImageNet)",
                    "ResNeXt(ImageNet)"
                ],
                "webp": [
                    "Alexnet(ImageNet)",
                    "SwinTransformer(ImageNet)",
                    "ConvNext(ImageNet)",
                    "DenseNet(ImageNet)",
                    "ResNeXt(ImageNet)"
                ],
            },
        },
        "inference_batch_config": {
            "Alexnet(ImageNet)": batch_size_m,
            "SwinTransformer(ImageNet)": batch_size_m,
            "ConvNext(ImageNet)": batch_size_m,
            "DenseNet(ImageNet)": batch_size_m,
            "ResNeXt(ImageNet)": batch_size_m
        },
        "attacker_config": {
            "DADD": {
                "clip_min": 0,
                "clip_max": 1,
                "attack_type": "UNTARGETED",
                "max_iterations": 10000,
                "spherical_step": 1e-2,
                "source_step_convergence": 1e-10,
                "step_adaptation": 1.5,
                "source_step": 1e-2,
                "update_stats_every_k": 10,
            },
        },
        "trans_config": {
        }
    }
    task_manager.init_task(task_token="test_DADD", show_logo=True, run_device="cuda")

    # 多数据库模式
    use_multi_database(mode=MultiDatabaseMode.SIMPLE)

    security_evaluation = SecurityEvaluation(example_config)

    security_evaluation.attack_full_test(use_img_file=False, use_raw_nparray_data=True)
    security_evaluation.trans_full_test(use_raw_nparray_data=True)
