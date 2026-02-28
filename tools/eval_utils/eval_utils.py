import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils, box_utils

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    wandb = None
    WANDB_AVAILABLE = False


def compute_pr_curves(det_annos, gt_annos_by_frame, class_names, iou_threshold=0.5):
    """
    Compute precision-recall curves per class using BEV-IoU matching.

    Args:
        det_annos: list of dicts with keys 'frame_id', 'name', 'score', 'boxes_lidar'
        gt_annos_by_frame: dict mapping frame_id -> {'name': np.array, 'gt_boxes': np.array (N,7)}
        class_names: list of class name strings
        iou_threshold: IoU threshold for a detection to be counted as TP

    Returns:
        dict mapping class_name -> {'precision', 'recall', 'scores', 'ap', 'total_gt'}
    """
    pr_data = {}
    for cls in class_names:
        all_scores = []
        all_tp = []
        total_gt = 0

        for anno in det_annos:
            frame_id = anno['frame_id']
            gt = gt_annos_by_frame.get(str(frame_id), {'name': np.array([]), 'gt_boxes': np.zeros((0, 7))})

            # GT boxes for this class
            gt_mask = gt['name'] == cls
            gt_boxes = gt['gt_boxes'][gt_mask]
            total_gt += int(gt_mask.sum())

            # Predicted boxes for this class
            pred_mask = anno['name'] == cls
            if pred_mask.sum() == 0:
                continue
            pred_boxes = anno['boxes_lidar'][pred_mask]
            pred_scores = anno['score'][pred_mask]

            if len(gt_boxes) == 0:
                all_scores.extend(pred_scores.tolist())
                all_tp.extend([0] * len(pred_scores))
                continue

            # BEV-IoU between predictions and GT (axis-aligned for speed)
            iou_matrix = box_utils.boxes3d_nearest_bev_iou(
                torch.from_numpy(pred_boxes).float(),
                torch.from_numpy(gt_boxes).float()
            ).numpy()  # (N_pred, N_gt)

            # Greedy TP assignment in descending score order
            score_order = np.argsort(-pred_scores)
            matched_gt = set()
            tp_flags = np.zeros(len(pred_boxes), dtype=np.int32)
            for idx in score_order:
                best_gt = int(np.argmax(iou_matrix[idx]))
                if iou_matrix[idx, best_gt] >= iou_threshold and best_gt not in matched_gt:
                    tp_flags[idx] = 1
                    matched_gt.add(best_gt)

            all_scores.extend(pred_scores.tolist())
            all_tp.extend(tp_flags.tolist())

        if len(all_scores) == 0 or total_gt == 0:
            pr_data[cls] = None
            continue

        order = np.argsort(-np.array(all_scores))
        tp_sorted = np.array(all_tp, dtype=np.float32)[order]
        scores_sorted = np.array(all_scores)[order]

        cum_tp = np.cumsum(tp_sorted)
        cum_fp = np.cumsum(1.0 - tp_sorted)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        recall = cum_tp / max(total_gt, 1)

        # Apply precision envelope (monotonically decreasing from right to left)
        # This is the standard VOC/KITTI interpolation: prec[i] = max(prec[i:])
        precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]

        # Compute AP via trapezoidal rule over the smoothed envelope
        ap = float(np.trapz(precision_envelope, recall)) if len(recall) > 1 else 0.0

        pr_data[cls] = {
            'precision': precision,
            'precision_envelope': precision_envelope,
            'recall': recall,
            'scores': scores_sorted,
            'ap': ap,
            'total_gt': total_gt,
        }

    return pr_data


def plot_pr_curves(pr_data, result_dir, epoch_id, logger):
    """Save per-class PR curve PNGs and raw data to result_dir/pr_curves/."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning('matplotlib not available; skipping PR curve plots')
        return None

    pr_dir = result_dir / 'pr_curves'
    pr_dir.mkdir(parents=True, exist_ok=True)

    classes = list(pr_data.keys())
    n_cls = len(classes)
    fig, axes = plt.subplots(1, n_cls, figsize=(5 * n_cls, 4), squeeze=False)
    axes = axes[0]

    for ax, cls in zip(axes, classes):
        data = pr_data[cls]
        if data is None:
            ax.text(0.5, 0.5, 'No detections', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(cls)
        else:
            ax.plot(data['recall'], data['precision'], linewidth=1, alpha=0.4, label='raw')
            ax.plot(data['recall'], data['precision_envelope'], linewidth=2, label='envelope')
            ax.legend(fontsize=8)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.4)
            ax.set_title(f"{cls}\nAP={data['ap']:.3f}  GT={data['total_gt']}")

    fig.suptitle(f'PR Curves — Epoch {epoch_id}  (BEV-IoU ≥ 0.5)')
    plt.tight_layout()

    fig_path = pr_dir / f'pr_curve_epoch_{epoch_id}.png'
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')

    # Save raw arrays for later analysis
    np.save(str(pr_dir / f'pr_data_epoch_{epoch_id}.npy'), pr_data, allow_pickle=True)

    logger.info('PR curves saved to %s' % fig_path)
    return fig  # caller is responsible for plt.close(fig) after wandb logging


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []
    gt_annos_by_frame = {}  # frame_id -> {'gt_boxes': (N,7), 'name': (N,)}

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):

        if cfg.get('TARGET_DATA_CONFIG', None) is not None:
            batch_dict['domain'] = 1

        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            try:
                pred_dicts, ret_dict = model(batch_dict)
            except:
                pred_dicts, ret_dict, _ = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos

        # Collect GT annotations for PR curve computation
        if 'gt_boxes' in batch_dict:
            gt_boxes_batch = batch_dict['gt_boxes'].cpu().numpy()  # (B, N, 8)
            shift_coor = cfg.get('TARGET_DATA_CONFIG', cfg.get('DATA_CONFIG', {})).get('SHIFT_COOR', None)
            for b in range(batch_dict['batch_size']):
                fid = str(batch_dict['frame_id'][b])
                gt_b = gt_boxes_batch[b]  # (N, 8)
                valid = gt_b[:, 7] > 0
                gt_b = gt_b[valid]
                gt_boxes_7 = gt_b[:, :7].copy()
                # Un-shift GT to match prediction frame (generate_prediction_dicts un-shifts preds)
                if shift_coor is not None and len(gt_boxes_7) > 0:
                    gt_boxes_7[:, 0:3] -= np.array(shift_coor, dtype=np.float32)
                gt_names = np.array(
                    [class_names[int(c) - 1] for c in gt_b[:, 7]]
                ) if len(gt_b) else np.array([], dtype=str)
                gt_annos_by_frame[fid] = {
                    'gt_boxes': gt_boxes_7,
                    'name': gt_names,
                }
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    print(dataloader.dataset.class_names)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))
    ret_dict['avg_pred_objects'] = total_pred_objects / max(1, len(det_annos))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    # --- PR curves ---
    pr_fig = None
    pr_data = None
    if gt_annos_by_frame:
        try:
            pr_data = compute_pr_curves(det_annos, gt_annos_by_frame, class_names, iou_threshold=0.5)
            pr_fig = plot_pr_curves(pr_data, result_dir, epoch_id, logger)
            for cls, data in pr_data.items():
                if data is not None:
                    ret_dict['pr/AP_%s' % cls] = data['ap']
        except Exception as e:
            logger.warning('PR curve generation failed: %s' % str(e))

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    if WANDB_AVAILABLE and wandb is not None and wandb.run is not None:
        wandb_log = {f'val/{key}': val for key, val in ret_dict.items()}
        wandb.define_metric('val/epoch')
        wandb.define_metric('val/*', step_metric='val/epoch')
        wandb_log['val/epoch'] = epoch_id

        # Log PR curve figure as image
        if pr_fig is not None:
            wandb_log['val/pr_curves'] = wandb.Image(pr_fig)

        # Log interactive per-class PR curves using wandb line plots
        if pr_data is not None:
            for cls, data in pr_data.items():
                if data is None:
                    continue
                # Subsample to max 500 points for wandb efficiency
                n = len(data['recall'])
                step = max(1, n // 500)
                recall_sub = data['recall'][::step].tolist()
                precision_sub = data['precision'][::step].tolist()
                pr_table = wandb.Table(
                    data=list(zip(recall_sub, precision_sub)),
                    columns=['recall', 'precision']
                )
                wandb_log[f'val/pr_table_{cls}'] = wandb.plot.line(
                    pr_table, 'recall', 'precision',
                    title=f'PR Curve — {cls} (epoch {epoch_id})'
                )

        wandb.log(wandb_log)

    # Clean up matplotlib figure
    if pr_fig is not None:
        import matplotlib.pyplot as plt
        plt.close(pr_fig)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
