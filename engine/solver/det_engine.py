# """
# DEIM: DETR with Improved Matching for Fast Convergence
# Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
# ---------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# """
#
#
# import sys
# import math
# # from typing import Iterable
# from ast import List # add swanlab
# from typing import Dict, Iterable # add swanlab
# import numpy as np
# # import swanlab # add swanlab
#
#
# import torch
# import torch.amp
# from torch.utils.tensorboard import SummaryWriter
# from torch.cuda.amp.grad_scaler import GradScaler
#
# from ..optim import ModelEMA, Warmup
# from ..data import CocoEvaluator, mscoco_category2label
# from ..misc import MetricLogger, SmoothedValue, dist_utils
# from .validator import Validator, scale_boxes # add swanlab
#
# import tqdm
#
#
# def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, total_epoch: int,
#                     use_swanlab: bool, max_norm: float = 0, verbose=True, **kwargs):
#     if use_swanlab: # add swanlab
#         import swanlab
#
#     model.train()
#     criterion.train()
#     metric_logger = MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = f'Epoch: [{epoch+1}/{total_epoch}]'
#     # header = 'Epoch: [{}]'.format(epoch)
#
#     print_freq = kwargs.get('print_freq', 10)
#     writer: SummaryWriter = kwargs.get('writer', None)
#
#     ema: ModelEMA = kwargs.get('ema', None)
#     scaler: GradScaler = kwargs.get('scaler', None)
#     lr_warmup_scheduler: Warmup = kwargs.get('lr_warmup_scheduler', None)
#
#     # --------------------------- add swanlab (train loss) ------------------------------
#     # loss_keys = ['mal', 'bbox', 'giou', 'fgl']
#     # loss_metrics = {f'train/{k}_loss': [] for k in loss_keys}
#     primary_loss_keys = []
#     loss_metrics = []
#     losses = []
#     # --------------------------- add swanlab (train loss) ------------------------------
#
#     cur_iters = epoch * len(data_loader)
#
#     # --------------------- tqdm ---------------------
#     if dist_utils.is_main_process():
#         pbar = tqdm.tqdm(total=len(data_loader), desc=header, dynamic_ncols=True, mininterval=0.1)
#     else:
#         pbar = None
#     for i, (samples, targets) in enumerate(data_loader):
#     # --------------------- tqdm ---------------------
#     # for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#         samples = samples.to(device)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#         global_step = epoch * len(data_loader) + i
#         metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))
#
#         if scaler is not None:
#             with torch.autocast(device_type=str(device), cache_enabled=True):
#                 outputs = model(samples, targets=targets)
#
#             if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
#                 print(outputs['pred_boxes'])
#                 state = model.state_dict()
#                 new_state = {}
#                 for key, value in model.state_dict().items():
#                     # Replace 'module' with 'model' in each key
#                     new_key = key.replace('module.', '')
#                     # Add the updated key-value pair to the state dictionary
#                     state[new_key] = value
#                 new_state['model'] = state
#                 dist_utils.save_on_master(new_state, "./NaN.pth")
#
#             with torch.autocast(device_type=str(device), enabled=False):
#                 loss_dict = criterion(outputs, targets, **metas)
#
#             loss = sum(loss_dict.values())
#             scaler.scale(loss).backward()
#
#             if max_norm > 0:
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad()
#
#         else:
#             outputs = model(samples, targets=targets)
#             loss_dict = criterion(outputs, targets, **metas)
#
#             loss : torch.Tensor = sum(loss_dict.values())
#             optimizer.zero_grad()
#             loss.backward()
#
#             if max_norm > 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#
#             optimizer.step()
#
#         # ema
#         if ema is not None:
#             ema.update(model)
#
#         if self_lr_scheduler:
#             optimizer = lr_scheduler.step(cur_iters + i, optimizer)
#         else:
#             if lr_warmup_scheduler is not None:
#                 lr_warmup_scheduler.step()
#
#         loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
#         loss_value = sum(loss_dict_reduced.values())
#
#         # --------------------------- add swanlab (train loss) ------------------------------
#         if i == 0:  # 只在第一次迭代时初始化
#             # 识别主要损失项：只包含一个下划线的键
#             primary_loss_keys = [k for k in loss_dict_reduced.keys() if k.count('_') == 1]
#             # 创建损失收集器
#             loss_metrics = {f'train_loss/{k.split("_")[-1]}_loss': [] for k in primary_loss_keys}
#
#         for k in primary_loss_keys:
#             if k in loss_dict_reduced:
#                 loss_metrics[f'train_loss/{k.split("_")[-1]}_loss'].append(loss_dict_reduced[k].item())
#             else:
#                 # 如果损失项不存在，记录为0（可能在某些阶段未激活）
#                 loss_metrics[f'train_loss/{k.split("_")[-1]}_loss'].append(0.0)
#         losses.append(loss_value.detach().cpu().numpy())
#         # --------------------------- add swanlab (train loss) ------------------------------
#
#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             print(loss_dict_reduced)
#             sys.exit(1)
#
#         metric_logger.update(loss=loss_value, **loss_dict_reduced)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#
#         # --------------------- tqdm ---------------------
#         if pbar is not None:
#             # 获取关键指标
#             if verbose:
#                 stats = {k: meter.median for k, meter in metric_logger.meters.items()
#                          if not k.startswith(('time', 'data')) and k.count('_') < 2}
#             else:
#                 stats = {k: meter.median for k, meter in metric_logger.meters.items()
#                          if not k.startswith(('time', 'data'))}
#
#             # 构建进度条信息
#             postfix = {}
#             for k, v in stats.items():
#                 if k == 'lr':
#                     postfix[k] = f'{v:.6f}'
#                 else:
#                     postfix[k] = f'{v:.4f}'
#
#             # 更新进度条显示
#             pbar.set_postfix(**postfix)
#             pbar.update(1)
#         # --------------------- tqdm ---------------------
#
#         if writer and dist_utils.is_main_process() and global_step % 10 == 0:
#             writer.add_scalar('Loss/total', loss_value.item(), global_step)
#             for j, pg in enumerate(optimizer.param_groups):
#                 writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
#             for k, v in loss_dict_reduced.items():
#                 writer.add_scalar(f'Loss/{k}', v.item(), global_step)
#
#     # --------------------------- add swanlab (train loss) ------------------------------
#     if use_swanlab:
#         avg_losses = {}
#         for k, v in loss_metrics.items():
#             avg_losses[k] = np.mean(v) if v else 0.0
#
#         log_data = {
#             "lr": optimizer.param_groups[0]["lr"],
#             "epoch": epoch,
#             "train_loss/total_loss": np.mean(losses)
#         }
#         log_data.update(avg_losses)
#         swanlab.log(log_data)
#     # --------------------------- add swanlab (train loss) ------------------------------
#
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     # --------------------- tqdm ---------------------
#     if pbar is not None:
#         pbar.close()
#     if dist_utils.is_main_process():
#         print("Averaged stats:", metric_logger)
#     # --------------------- tqdm ---------------------
#     # print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
#
#
# @torch.no_grad()
# def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader,
#              coco_evaluator: CocoEvaluator, device,
#              epoch: int, use_swanlab: bool, total_epochs: int):
#     if use_swanlab:
#         import swanlab
#         import matplotlib.pyplot as plt
#         import numpy as np
#         from pathlib import Path
#         from PIL import Image as PILImage
#         from collections import defaultdict
#
#     model.eval()
#     criterion.eval()
#     # 修复：处理coco_evaluator为None的情况
#     if coco_evaluator is not None:
#         coco_evaluator.cleanup()
#
#     metric_logger = MetricLogger(delimiter="  ")
#     header = 'Test:'
#
#     # 修复：处理coco_evaluator为None的情况
#     iou_types = coco_evaluator.iou_types if coco_evaluator is not None else []
#
#     # 存储用于验证器的数据
#     gt_for_validator: List[Dict[str, torch.Tensor]] = []
#     preds_for_validator: List[Dict[str, torch.Tensor]] = []
#
#     # 存储COCO格式的预测结果
#     coco_results = []
#
#     if dist_utils.is_main_process():
#         pbar = tqdm.tqdm(total=len(data_loader), desc=header, position=0,
#                          dynamic_ncols=True, mininterval=0.5)
#     else:
#         pbar = None
#
#     for samples, targets in data_loader:
#         samples = samples.to(device)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
#         with torch.no_grad():
#             outputs = model(samples)
#             orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
#             results = postprocessor(outputs, orig_target_sizes)
#
#         res = {target['image_id'].item(): output for target, output in zip(targets, results)}
#         if coco_evaluator is not None:
#             coco_evaluator.update(res)
#
#         # 收集COCO格式的预测结果
#         for target, output in zip(targets, results):
#             image_id = target["image_id"].item()
#             boxes = output["boxes"]
#             scores = output["scores"]
#             labels = output["labels"]
#
#             # 转换为COCO格式
#             for box, score, label in zip(boxes, scores, labels):
#                 box = box.cpu().numpy().tolist()
#                 coco_results.append({
#                     "image_id": image_id,
#                     "category_id": label.item(),
#                     "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],  # xywh
#                     "score": score.item()
#                 })
#
#         # 存储用于验证器的数据 - 修复：移除GT框缩放操作
#         for idx, (target, result) in enumerate(zip(targets, results)):
#             # 直接使用原始GT框，不进行缩放
#             gt_for_validator.append({
#                 "boxes": target["boxes"],
#                 "labels": target["labels"],
#             })
#
#             preds_for_validator.append({
#                 "boxes": result["boxes"],
#                 "labels": result["labels"],
#                 "scores": result["scores"]
#             })
#
#         if pbar is not None:
#             pbar.update(1)
#
#     if pbar is not None:
#         pbar.close()
#
#     # 验证器计算
#     validator = Validator(gt_for_validator, preds_for_validator)
#     metrics = validator.compute_metrics()
#     print("Val_Metrics:", metrics)
#
#     # ============== 修复PR曲线计算 ==============
#     if use_swanlab and dist_utils.is_main_process() and epoch == total_epochs - 1:
#         # 确保COCO评估器已经完成计算
#         if coco_evaluator is not None:
#             coco_evaluator.synchronize_between_processes()
#             coco_evaluator.accumulate()
#             coco_evaluator.summarize()
#
#             # 获取bbox评估器
#             if 'bbox' in coco_evaluator.coco_eval:
#                 coco_eval = coco_evaluator.coco_eval['bbox']
#             else:
#                 print("Warning: No bbox evaluator found, skipping PR curve")
#                 coco_eval = None
#         else:
#             coco_eval = None
#
#         if coco_eval is not None:
#             # 确保评估器已经计算了结果
#             if not hasattr(coco_eval, 'eval'):
#                 print("COCO evaluator has not been evaluated, running accumulate now")
#                 coco_eval.accumulate()
#
#             # 检查eval属性是否存在
#             if hasattr(coco_eval, 'eval') and 'precision' in coco_eval.eval:
#                 # 获取所有类别信息
#                 coco_gt = coco_eval.cocoGt
#                 cat_ids = coco_gt.getCatIds()
#                 class_names = {}
#                 class_ap_report = {}
#
#                 # 提取PR数据
#                 precision = coco_eval.eval['precision']
#
#                 # 获取精度数组的形状信息
#                 t_idx = 0  # IoU=0.5的索引
#                 a_idx = 0  # 所有面积范围
#                 m_idx = -1  # 最大检测数=100
#
#                 # 检查维度
#                 if len(precision.shape) >= 5:
#                     pr_data = precision[t_idx, :, :, a_idx, m_idx]  # 形状: [101, 类别数]
#                 elif len(precision.shape) == 4:
#                     pr_data = precision[t_idx, :, :, a_idx]  # 形状: [101, 类别数]
#                 else:
#                     print(f"Warning: Unexpected precision shape {precision.shape}, using first dimension")
#                     pr_data = precision[0] if len(precision) > 0 else None
#
#                 if pr_data is None or pr_data.size == 0:
#                     print("Warning: No precision data available for PR curve")
#                 else:
#                     # 创建101个召回率点
#                     px = np.linspace(0, 1, 101)
#
#                     # 计算每类别的AP值
#                     ap_values = []
#                     py_per_class = []
#
#                     num_classes = pr_data.shape[1] if len(pr_data.shape) > 1 else 0
#
#                     for cat_idx in range(num_classes):
#                         if cat_idx < len(cat_ids):
#                             cat_id = cat_ids[cat_idx]
#                         else:
#                             continue
#
#                         cat_data = coco_gt.loadCats(cat_id)[0]
#                         class_name = cat_data['name']
#                         class_names[cat_id] = class_name
#
#                         cat_precision = pr_data[:, cat_idx]
#
#                         # 过滤无效值
#                         valid_mask = cat_precision > -1
#                         valid_precision = cat_precision[valid_mask]
#
#                         ap = np.mean(valid_precision) if len(valid_precision) > 0 else 0.0
#                         ap_values.append(ap)
#                         class_ap_report[f"AP50_{class_name}"] = ap
#
#                         # 存储曲线数据用于绘图
#                         py_per_class.append(cat_precision)
#
#                     # 计算平均AP
#                     mean_ap = np.mean(ap_values) if ap_values else 0.0
#                     class_ap_report["mAP50"] = mean_ap
#
#                     # 计算平均精确率曲线
#                     valid_pr_data = np.where(pr_data == -1, 0, pr_data)
#                     mean_py = np.mean(valid_pr_data, axis=1)
#
#                     # 绘制PR曲线
#                     fig, ax = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)
#
#                     # 设置颜色循环
#                     colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(py_per_class))))
#
#                     # 绘制各类别曲线
#                     max_classes_to_plot = min(8, len(py_per_class))  # 限制显示的类别数量
#                     for i in range(max_classes_to_plot):
#                         cat_precision = py_per_class[i]
#                         cat_id = cat_ids[i]
#                         color = colors[i % len(colors)]
#                         ap_value = ap_values[i] if i < len(ap_values) else 0.0
#
#                         ax.plot(px, cat_precision, linewidth=1.5, alpha=0.7, color=color,
#                                 label=f"{class_names[cat_id]} AP={ap_value:.3f}")
#
#                     # 绘制平均曲线
#                     ax.plot(px, mean_py, linewidth=3, color="blue", linestyle='-',
#                             label=f"mAP@0.5 = {mean_ap:.3f}")
#
#                     # 设置图表属性
#                     ax.set_xlabel("Recall", fontsize=12)
#                     ax.set_ylabel("Precision", fontsize=12)
#                     ax.set_xlim(0, 1)
#                     ax.set_ylim(0, 1.05)
#                     ax.grid(True, linestyle='--', alpha=0.7)
#                     ax.set_title(f"Precision-Recall Curve @ IoU=0.5 (Epoch {epoch + 1}/{total_epochs})", fontsize=14)
#
#                     # 优化图例
#                     ax.legend(loc='lower left', bbox_to_anchor=(1.05, 0), ncol=1, fontsize=10)
#
#                     # 创建临时文件保存图像
#                     save_dir = Path("pr_curves")
#                     save_dir.mkdir(exist_ok=True)
#                     save_path = save_dir / f"pr_curve_epoch_{epoch + 1}.png"
#                     plt.savefig(save_path, dpi=250, bbox_inches='tight')
#                     plt.close(fig)
#
#                     # 转换为PIL图像并记录到SwanLab
#                     pil_image = PILImage.open(save_path)
#                     swanlab.log({
#                         "PR Curve": swanlab.Image(
#                             pil_image,
#                             caption=f"Precision-Recall Curve @ IoU=0.5 (Epoch {epoch + 1}, mAP={mean_ap:.3f})"
#                         )
#                     })
#
#                     # 记录AP值
#                     swanlab.log(class_ap_report)
#
#                     # 获取COCO评估器的AP50进行比较
#                     if hasattr(coco_eval, 'stats') and len(coco_eval.stats) > 1:
#                         coco_ap50 = coco_eval.stats[1]  # 索引1是AP@0.5
#                         print(f"COCO AP50: {coco_ap50:.4f}, Custom AP50: {mean_ap:.4f}")
#                         print(f"差异: {abs(coco_ap50 - mean_ap):.6f}")
#                     else:
#                         print("无法获取COCO评估器的AP50值")
#             else:
#                 print("Warning: COCO evaluator does not have 'precision' in eval dict")
#         else:
#             print("Skipping PR curve due to missing COCO evaluator")
#     # ============== PR曲线计算结束 ==============
#
#     # 初始化0点 - 仅在第一个epoch时记录0值
#     if use_swanlab and dist_utils.is_main_process() and epoch == 0:
#         # 创建所有指标的0值初始化点
#         zero_metrics = {f"val_metrics/{k}": 0.0 for k in metrics.keys()}
#         zero_metrics["epoch"] = 0  # 记录在0点位置
#         swanlab.log(zero_metrics)
#
#     # 记录实际指标值
#     if use_swanlab:
#         # 记录验证指标时，使用 epoch+1 作为横坐标值
#         log_dict = {f"val_metrics/{k}": v for k, v in metrics.items()}
#         log_dict["epoch"] = epoch + 1  # 从1开始记录
#         swanlab.log(log_dict)
#
#     # 处理COCO评估器
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#
#     # 修复：处理coco_evaluator为None的情况
#     if coco_evaluator is not None:
#         coco_evaluator.synchronize_between_processes()
#         coco_evaluator.accumulate()
#         coco_evaluator.summarize()
#
#     stats = {}
#     if coco_evaluator is not None:
#         if 'bbox' in iou_types:
#             stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
#         if 'segm' in iou_types:
#             stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
#
#     # 修复：处理coco_evaluator为None的情况
#     if coco_evaluator is not None and 'bbox' in coco_evaluator.coco_eval:
#         return stats, coco_evaluator, coco_evaluator.coco_eval['bbox'].stats_as_dict
#     else:
#         return stats, coco_evaluator, None
# # @torch.no_grad()
# # def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader,
# #              coco_evaluator: CocoEvaluator, device,
# #              epoch: int, use_swanlab: bool, total_epochs: int):
# #     if use_swanlab:
# #         import swanlab
# #         import matplotlib.pyplot as plt
# #         import numpy as np
# #         from pathlib import Path
# #         from PIL import Image as PILImage
# #         from collections import defaultdict
# #
# #     model.eval()
# #     criterion.eval()
# #     coco_evaluator.cleanup()
# #
# #     metric_logger = MetricLogger(delimiter="  ")
# #     header = 'Test:'
# #
# #     iou_types = coco_evaluator.iou_types
# #
# #     # 存储用于验证器的数据
# #     gt_for_validator: List[Dict[str, torch.Tensor]] = []
# #     preds_for_validator: List[Dict[str, torch.Tensor]] = []
# #
# #     # 存储COCO格式的预测结果
# #     coco_results = []
# #
# #     if dist_utils.is_main_process():
# #         pbar = tqdm.tqdm(total=len(data_loader), desc=header, position=0,
# #                          dynamic_ncols=True, mininterval=0.5)
# #     else:
# #         pbar = None
# #
# #     for samples, targets in data_loader:
# #         samples = samples.to(device)
# #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
# #
# #         with torch.no_grad():
# #             outputs = model(samples)
# #             orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
# #             results = postprocessor(outputs, orig_target_sizes)
# #
# #         res = {target['image_id'].item(): output for target, output in zip(targets, results)}
# #         if coco_evaluator is not None:
# #             coco_evaluator.update(res)
# #
# #         # 收集COCO格式的预测结果
# #         for target, output in zip(targets, results):
# #             image_id = target["image_id"].item()
# #             boxes = output["boxes"]
# #             scores = output["scores"]
# #             labels = output["labels"]
# #
# #             # 转换为COCO格式
# #             for box, score, label in zip(boxes, scores, labels):
# #                 box = box.cpu().numpy().tolist()
# #                 coco_results.append({
# #                     "image_id": image_id,
# #                     "category_id": label.item(),
# #                     "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],  # xywh
# #                     "score": score.item()
# #                 })
# #
# #         # 存储用于验证器的数据
# #         for idx, (target, result) in enumerate(zip(targets, results)):
# #             scaled_boxes = scale_boxes(
# #                 torch.tensor(target["boxes"].tolist()).to(device),
# #                 (target["orig_size"][1], target["orig_size"][0]),
# #                 (samples[idx].shape[-1], samples[idx].shape[-2]),
# #             )
# #
# #             gt_for_validator.append({
# #                 "boxes": scaled_boxes,
# #                 "labels": target["labels"],
# #             })
# #
# #             preds_for_validator.append({
# #                 "boxes": result["boxes"],
# #                 "labels": result["labels"],
# #                 "scores": result["scores"]
# #             })
# #
# #         if pbar is not None:
# #             pbar.update(1)
# #
# #     if pbar is not None:
# #         pbar.close()
# #
# #     # 验证器计算
# #     validator = Validator(gt_for_validator, preds_for_validator)
# #     metrics = validator.compute_metrics()
# #     print("Val_Metrics:", metrics)
# #
# #     # ============== 修复PR曲线计算 ==============
# #     if use_swanlab and dist_utils.is_main_process() and epoch == total_epochs - 1:
# #         # 确保COCO评估器已经完成计算
# #         if coco_evaluator is not None:
# #             coco_evaluator.synchronize_between_processes()
# #             coco_evaluator.accumulate()
# #             coco_evaluator.summarize()
# #
# #             # 获取bbox评估器
# #             if 'bbox' in coco_evaluator.coco_eval:
# #                 coco_eval = coco_evaluator.coco_eval['bbox']
# #             else:
# #                 print("Warning: No bbox evaluator found, skipping PR curve")
# #                 coco_eval = None
# #         else:
# #             coco_eval = None
# #
# #         if coco_eval is not None:
# #             # 确保评估器已经计算了结果
# #             if not hasattr(coco_eval, 'eval'):
# #                 print("COCO evaluator has not been evaluated, running accumulate now")
# #                 coco_eval.accumulate()
# #
# #             # 检查eval属性是否存在
# #             if hasattr(coco_eval, 'eval') and 'precision' in coco_eval.eval:
# #                 # 获取所有类别信息
# #                 coco_gt = coco_eval.cocoGt
# #                 cat_ids = coco_gt.getCatIds()
# #                 class_names = {}
# #                 class_ap_report = {}
# #
# #                 # 提取PR数据
# #                 precision = coco_eval.eval['precision']
# #
# #                 # 获取精度数组的形状信息
# #                 t_idx = 0  # IoU=0.5的索引
# #                 a_idx = 0  # 所有面积范围
# #                 m_idx = -1  # 最大检测数=100
# #
# #                 # 检查维度
# #                 if len(precision.shape) >= 5:
# #                     pr_data = precision[t_idx, :, :, a_idx, m_idx]  # 形状: [101, 类别数]
# #                 elif len(precision.shape) == 4:
# #                     pr_data = precision[t_idx, :, :, a_idx]  # 形状: [101, 类别数]
# #                 else:
# #                     print(f"Warning: Unexpected precision shape {precision.shape}, using first dimension")
# #                     pr_data = precision[0] if len(precision) > 0 else None
# #
# #                 if pr_data is None or pr_data.size == 0:
# #                     print("Warning: No precision data available for PR curve")
# #                 else:
# #                     # 创建101个召回率点
# #                     px = np.linspace(0, 1, 101)
# #
# #                     # 计算每类别的AP值
# #                     ap_values = []
# #                     py_per_class = []
# #
# #                     num_classes = pr_data.shape[1] if len(pr_data.shape) > 1 else 0
# #
# #                     for cat_idx in range(num_classes):
# #                         if cat_idx < len(cat_ids):
# #                             cat_id = cat_ids[cat_idx]
# #                         else:
# #                             continue
# #
# #                         cat_data = coco_gt.loadCats(cat_id)[0]
# #                         class_name = cat_data['name']
# #                         class_names[cat_id] = class_name
# #
# #                         cat_precision = pr_data[:, cat_idx]
# #
# #                         # 过滤无效值
# #                         valid_mask = cat_precision > -1
# #                         valid_precision = cat_precision[valid_mask]
# #
# #                         ap = np.mean(valid_precision) if len(valid_precision) > 0 else 0.0
# #                         ap_values.append(ap)
# #                         class_ap_report[f"AP50_{class_name}"] = ap
# #
# #                         # 存储曲线数据用于绘图
# #                         py_per_class.append(cat_precision)
# #
# #                     # 计算平均AP
# #                     mean_ap = np.mean(ap_values) if ap_values else 0.0
# #                     class_ap_report["mAP50"] = mean_ap
# #
# #                     # 计算平均精确率曲线
# #                     valid_pr_data = np.where(pr_data == -1, 0, pr_data)
# #                     mean_py = np.mean(valid_pr_data, axis=1)
# #
# #                     # 绘制PR曲线
# #                     fig, ax = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)
# #
# #                     # 设置颜色循环
# #                     colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(py_per_class))))
# #
# #                     # 绘制各类别曲线
# #                     max_classes_to_plot = min(8, len(py_per_class))  # 限制显示的类别数量
# #                     for i in range(max_classes_to_plot):
# #                         cat_precision = py_per_class[i]
# #                         cat_id = cat_ids[i]
# #                         color = colors[i % len(colors)]
# #                         ap_value = ap_values[i] if i < len(ap_values) else 0.0
# #
# #                         ax.plot(px, cat_precision, linewidth=1.5, alpha=0.7, color=color,
# #                                 label=f"{class_names[cat_id]} AP={ap_value:.3f}")
# #
# #                     # 绘制平均曲线
# #                     ax.plot(px, mean_py, linewidth=3, color="blue", linestyle='-',
# #                             label=f"mAP@0.5 = {mean_ap:.3f}")
# #
# #                     # 设置图表属性
# #                     ax.set_xlabel("Recall", fontsize=12)
# #                     ax.set_ylabel("Precision", fontsize=12)
# #                     ax.set_xlim(0, 1)
# #                     ax.set_ylim(0, 1.05)
# #                     ax.grid(True, linestyle='--', alpha=0.7)
# #                     ax.set_title(f"Precision-Recall Curve @ IoU=0.5 (Epoch {epoch + 1}/{total_epochs})", fontsize=14)
# #
# #                     # 优化图例
# #                     ax.legend(loc='lower left', bbox_to_anchor=(1.05, 0), ncol=1, fontsize=10)
# #
# #                     # 创建临时文件保存图像
# #                     save_dir = Path("pr_curves")
# #                     save_dir.mkdir(exist_ok=True)
# #                     save_path = save_dir / f"pr_curve_epoch_{epoch + 1}.png"
# #                     plt.savefig(save_path, dpi=250, bbox_inches='tight')
# #                     plt.close(fig)
# #
# #                     # 转换为PIL图像并记录到SwanLab
# #                     pil_image = PILImage.open(save_path)
# #                     swanlab.log({
# #                         "PR Curve": swanlab.Image(
# #                             pil_image,
# #                             caption=f"Precision-Recall Curve @ IoU=0.5 (Epoch {epoch + 1}, mAP={mean_ap:.3f})"
# #                         )
# #                     })
# #
# #                     # 记录AP值
# #                     swanlab.log(class_ap_report)
# #
# #                     # 获取COCO评估器的AP50进行比较
# #                     if hasattr(coco_eval, 'stats') and len(coco_eval.stats) > 1:
# #                         coco_ap50 = coco_eval.stats[1]  # 索引1是AP@0.5
# #                         print(f"COCO AP50: {coco_ap50:.4f}, Custom AP50: {mean_ap:.4f}")
# #                         print(f"差异: {abs(coco_ap50 - mean_ap):.6f}")
# #                     else:
# #                         print("无法获取COCO评估器的AP50值")
# #             else:
# #                 print("Warning: COCO evaluator does not have 'precision' in eval dict")
# #         else:
# #             print("Skipping PR curve due to missing COCO evaluator")
# #     # ============== PR曲线计算结束 ==============
# #
# #     if use_swanlab:
# #         metrics = {f"val_metrics/{k}": v for k, v in metrics.items()}
# #         metrics["epoch"] = epoch
# #         swanlab.log(metrics)
# #
# #     # 处理COCO评估器
# #     metric_logger.synchronize_between_processes()
# #     print("Averaged stats:", metric_logger)
# #     if coco_evaluator is not None:
# #         coco_evaluator.synchronize_between_processes()
# #         coco_evaluator.accumulate()
# #         coco_evaluator.summarize()
# #
# #     stats = {}
# #     if coco_evaluator is not None:
# #         if 'bbox' in iou_types:
# #             stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
# #         if 'segm' in iou_types:
# #             stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
# #
# #     return stats, coco_evaluator, coco_evaluator.coco_eval['bbox'].stats_as_dict
#
#
#
"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import sys
import math
# from typing import Iterable
from ast import List # add swanlab
from typing import Dict, Iterable # add swanlab
import numpy as np
# import swanlab # add swanlab


import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator, mscoco_category2label
from ..misc import MetricLogger, SmoothedValue, dist_utils
from .validator import Validator, scale_boxes # add swanlab

import tqdm

def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, total_epoch: int,
                    use_swanlab: bool, max_norm: float = 0, verbose=True, **kwargs):
    if use_swanlab: # add swanlab
        import swanlab

    model.train()
    criterion.train()
    # >>> 新增：清空前一 epoch 的缓存
    for m in model.modules():
        if hasattr(m, 'current_epoch_scales'):
            m.current_epoch_scales.clear()
            m.current_epoch_gates.clear()
    # <<<
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch+1}/{total_epoch}]'
    # header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer: SummaryWriter = kwargs.get('writer', None)

    ema: ModelEMA = kwargs.get('ema', None)
    scaler: GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler: Warmup = kwargs.get('lr_warmup_scheduler', None)

    # --------------------------- add swanlab (train loss) ------------------------------
    # loss_keys = ['mal', 'bbox', 'giou', 'fgl']
    # loss_metrics = {f'train/{k}_loss': [] for k in loss_keys}
    primary_loss_keys = []
    loss_metrics = []
    losses = []
    # --------------------------- add swanlab (train loss) ------------------------------

    cur_iters = epoch * len(data_loader)

    # --------------------- tqdm ---------------------
    if dist_utils.is_main_process():
        pbar = tqdm.tqdm(total=len(data_loader), desc=header, dynamic_ncols=True, mininterval=0.1)
    else:
        pbar = None
    for i, (samples, targets) in enumerate(data_loader):
    # --------------------- tqdm ---------------------
    # for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace('module.', '')
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        # --------------------------- add swanlab (train loss) ------------------------------
        if i == 0:  # 只在第一次迭代时初始化
            # 识别主要损失项：只包含一个下划线的键
            primary_loss_keys = [k for k in loss_dict_reduced.keys() if k.count('_') == 1]
            # 创建损失收集器
            loss_metrics = {f'train_loss/{k.split("_")[-1]}_loss': [] for k in primary_loss_keys}

        for k in primary_loss_keys:
            if k in loss_dict_reduced:
                loss_metrics[f'train_loss/{k.split("_")[-1]}_loss'].append(loss_dict_reduced[k].item())
            else:
                # 如果损失项不存在，记录为0（可能在某些阶段未激活）
                loss_metrics[f'train_loss/{k.split("_")[-1]}_loss'].append(0.0)
        losses.append(loss_value.detach().cpu().numpy())
        # --------------------------- add swanlab (train loss) ------------------------------

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # --------------------- tqdm ---------------------
        if pbar is not None:
            # 获取关键指标
            if verbose:
                stats = {k: meter.median for k, meter in metric_logger.meters.items()
                         if not k.startswith(('time', 'data')) and k.count('_') < 2}
            else:
                stats = {k: meter.median for k, meter in metric_logger.meters.items()
                         if not k.startswith(('time', 'data'))}

            # 构建进度条信息
            postfix = {}
            for k, v in stats.items():
                if k == 'lr':
                    postfix[k] = f'{v:.6f}'
                else:
                    postfix[k] = f'{v:.4f}'

            # 更新进度条显示
            pbar.set_postfix(**postfix)
            pbar.update(1)
        # --------------------- tqdm ---------------------

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    # --------------------------- add swanlab (train loss) ------------------------------
    if use_swanlab:
        avg_losses = {}
        for k, v in loss_metrics.items():
            avg_losses[k] = np.mean(v) if v else 0.0

        log_data = {
            "lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch,
            "train_loss/total_loss": np.mean(losses)
        }
        log_data.update(avg_losses)

        # >>> 新增：收集所有 gate 模块的 scale/gate 均值
        gate_stats = {}
        for name, m in model.named_modules():
            if hasattr(m, 'current_epoch_scales') and m.current_epoch_scales:
                scales = torch.cat(m.current_epoch_scales).cpu()
                gates = torch.cat(m.current_epoch_gates).cpu()
                prefix = f"gate/{name.replace('.', '_')}"
                gate_stats[f"{prefix}/scale_mean"] = scales.mean().item()
                gate_stats[f"{prefix}/gate_mean"] = gates.mean().item()
                # 可选：加 std/min/max
                # gate_stats[f"{prefix}/scale_std"] = scales.std().item()
        log_data.update(gate_stats)
        # <<<

        swanlab.log(log_data)
    # --------------------------- add swanlab (train loss) ------------------------------

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # --------------------- tqdm ---------------------
    if pbar is not None:
        pbar.close()
    if dist_utils.is_main_process():
        print("Averaged stats:", metric_logger)
    # --------------------- tqdm ---------------------
    # print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader,
             coco_evaluator: CocoEvaluator, device,
             epoch: int, use_swanlab: bool, total_epochs: int):
    if use_swanlab:
        import swanlab
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path
        from PIL import Image as PILImage
        from collections import defaultdict

    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = coco_evaluator.iou_types

    # 存储用于验证器的数据
    gt_for_validator: List[Dict[str, torch.Tensor]] = []
    preds_for_validator: List[Dict[str, torch.Tensor]] = []

    # 存储COCO格式的预测结果
    coco_results = []

    if dist_utils.is_main_process():
        pbar = tqdm.tqdm(total=len(data_loader), desc=header, position=0,
                         dynamic_ncols=True, mininterval=0.5)
    else:
        pbar = None

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(samples)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # 收集COCO格式的预测结果
        for target, output in zip(targets, results):
            image_id = target["image_id"].item()
            boxes = output["boxes"]
            scores = output["scores"]
            labels = output["labels"]

            # 转换为COCO格式
            for box, score, label in zip(boxes, scores, labels):
                box = box.cpu().numpy().tolist()
                coco_results.append({
                    "image_id": image_id,
                    "category_id": label.item(),
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],  # xywh
                    "score": score.item()
                })

        # 存储用于验证器的数据
        for idx, (target, result) in enumerate(zip(targets, results)):
            scaled_boxes = scale_boxes(
                torch.tensor(target["boxes"].tolist()).to(device),
                (target["orig_size"][1], target["orig_size"][0]),
                (samples[idx].shape[-1], samples[idx].shape[-2]),
            )

            gt_for_validator.append({
                "boxes": scaled_boxes,
                "labels": target["labels"],
            })

            preds_for_validator.append({
                "boxes": result["boxes"],
                "labels": result["labels"],
                "scores": result["scores"]
            })

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    # 验证器计算
    validator = Validator(gt_for_validator, preds_for_validator)
    metrics = validator.compute_metrics()
    print("Val_Metrics:", metrics)

    # ============== 修复PR曲线计算 ==============
    if use_swanlab and dist_utils.is_main_process() and epoch == total_epochs - 1:
        # 确保COCO评估器已经完成计算
        if coco_evaluator is not None:
            coco_evaluator.synchronize_between_processes()
            coco_evaluator.accumulate()
            coco_evaluator.summarize()

            # 获取bbox评估器
            if 'bbox' in coco_evaluator.coco_eval:
                coco_eval = coco_evaluator.coco_eval['bbox']
            else:
                print("Warning: No bbox evaluator found, skipping PR curve")
                coco_eval = None
        else:
            coco_eval = None

        if coco_eval is not None:
            # 确保评估器已经计算了结果
            if not hasattr(coco_eval, 'eval'):
                print("COCO evaluator has not been evaluated, running accumulate now")
                coco_eval.accumulate()

            # 检查eval属性是否存在
            if hasattr(coco_eval, 'eval') and 'precision' in coco_eval.eval:
                # 获取所有类别信息
                coco_gt = coco_eval.cocoGt
                cat_ids = coco_gt.getCatIds()
                class_names = {}
                class_ap_report = {}

                # 提取PR数据
                precision = coco_eval.eval['precision']

                # 获取精度数组的形状信息
                t_idx = 0  # IoU=0.5的索引
                a_idx = 0  # 所有面积范围
                m_idx = -1  # 最大检测数=100

                # 检查维度
                if len(precision.shape) >= 5:
                    pr_data = precision[t_idx, :, :, a_idx, m_idx]  # 形状: [101, 类别数]
                elif len(precision.shape) == 4:
                    pr_data = precision[t_idx, :, :, a_idx]  # 形状: [101, 类别数]
                else:
                    print(f"Warning: Unexpected precision shape {precision.shape}, using first dimension")
                    pr_data = precision[0] if len(precision) > 0 else None

                if pr_data is None or pr_data.size == 0:
                    print("Warning: No precision data available for PR curve")
                else:
                    # 创建101个召回率点
                    px = np.linspace(0, 1, 101)

                    # 计算每类别的AP值
                    ap_values = []
                    py_per_class = []

                    num_classes = pr_data.shape[1] if len(pr_data.shape) > 1 else 0

                    for cat_idx in range(num_classes):
                        if cat_idx < len(cat_ids):
                            cat_id = cat_ids[cat_idx]
                        else:
                            continue

                        cat_data = coco_gt.loadCats(cat_id)[0]
                        class_name = cat_data['name']
                        class_names[cat_id] = class_name

                        cat_precision = pr_data[:, cat_idx]

                        # 过滤无效值
                        valid_mask = cat_precision > -1
                        valid_precision = cat_precision[valid_mask]

                        ap = np.mean(valid_precision) if len(valid_precision) > 0 else 0.0
                        ap_values.append(ap)
                        class_ap_report[f"AP50_{class_name}"] = ap

                        # 存储曲线数据用于绘图
                        py_per_class.append(cat_precision)

                    # 计算平均AP
                    mean_ap = np.mean(ap_values) if ap_values else 0.0
                    class_ap_report["mAP50"] = mean_ap

                    # 计算平均精确率曲线
                    valid_pr_data = np.where(pr_data == -1, 0, pr_data)
                    mean_py = np.mean(valid_pr_data, axis=1)

                    # 绘制PR曲线
                    fig, ax = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)

                    # 设置颜色循环
                    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(py_per_class))))

                    # 绘制各类别曲线
                    max_classes_to_plot = min(9, len(py_per_class))  # 限制显示的类别数量
                    for i in range(max_classes_to_plot):
                        cat_precision = py_per_class[i]
                        cat_id = cat_ids[i]
                        color = colors[i % len(colors)]
                        ap_value = ap_values[i] if i < len(ap_values) else 0.0

                        ax.plot(px, cat_precision, linewidth=1.5, alpha=0.7, color=color,
                                label=f"{class_names[cat_id]} AP={ap_value:.3f}")

                    # 绘制平均曲线
                    ax.plot(px, mean_py, linewidth=3, color="blue", linestyle='-',
                            label=f"mAP@0.5 = {mean_ap:.3f}")

                    # 设置图表属性
                    ax.set_xlabel("Recall", fontsize=12)
                    ax.set_ylabel("Precision", fontsize=12)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1.05)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.set_title(f"Precision-Recall Curve @ IoU=0.5 (Epoch {epoch + 1}/{total_epochs})", fontsize=14)

                    # 优化图例
                    ax.legend(loc='lower left', bbox_to_anchor=(1.05, 0), ncol=1, fontsize=10)

                    # 创建临时文件保存图像
                    save_dir = Path("pr_curves")
                    save_dir.mkdir(exist_ok=True)
                    save_path = save_dir / f"pr_curve_epoch_{epoch + 1}.png"
                    plt.savefig(save_path, dpi=250, bbox_inches='tight')
                    plt.close(fig)

                    # 转换为PIL图像并记录到SwanLab
                    pil_image = PILImage.open(save_path)
                    swanlab.log({
                        "PR Curve": swanlab.Image(
                            pil_image,
                            caption=f"Precision-Recall Curve @ IoU=0.5 (Epoch {epoch + 1}, mAP={mean_ap:.3f})"
                        )
                    })

                    # 记录AP值
                    swanlab.log(class_ap_report)

                    # 获取COCO评估器的AP50进行比较
                    if hasattr(coco_eval, 'stats') and len(coco_eval.stats) > 1:
                        coco_ap50 = coco_eval.stats[1]  # 索引1是AP@0.5
                        print(f"COCO AP50: {coco_ap50:.4f}, Custom AP50: {mean_ap:.4f}")
                        print(f"差异: {abs(coco_ap50 - mean_ap):.6f}")
                    else:
                        print("无法获取COCO评估器的AP50值")
            else:
                print("Warning: COCO evaluator does not have 'precision' in eval dict")
        else:
            print("Skipping PR curve due to missing COCO evaluator")
    # ============== PR曲线计算结束 ==============

    if use_swanlab:
        metrics = {f"val_metrics/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = epoch
        swanlab.log(metrics)

    # 处理COCO评估器
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator, coco_evaluator.coco_eval['bbox'].stats_as_dict
