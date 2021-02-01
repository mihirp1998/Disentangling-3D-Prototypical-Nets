import numpy as np
from utils_eval import makeDict
from nms_gpu import rotate_iou_gpu_eval
import ipdb
st = ipdb.set_trace
class MapScoreCalculator:
    def __init__(self):
        self.EPS = 1e-6

    def get_mAP(self, boxes_e, scores_e, boxes_g, scores_g):
        boxes_e = boxes_e.numpy()
        scores_e = scores_e.numpy()
        boxes_g = boxes_g.numpy()
        scores_g = scores_g.numpy()
        B, N, D = list(boxes_e.shape)
        map3d_e = self.eval_wrapper(boxes_g, boxes_e, scores_e)
        map3d_g = self.eval_wrapper(boxes_g+np.random.uniform(0.0, 0.01, []), boxes_g, scores_g)
        return map3d_e, map3d_g

    def eval_wrapper(self, boxes_g, boxes_e, scores_e):
        B, N, _ = boxes_e.shape
        B, N2, _ = boxes_g.shape
        
        if N > N2:
            padding = N-N2
            boxes_e = np.pad(boxes_e, ((0,0), (0,padding), (0,0)),
                             mode='constant')
            scores_e = np.pad(scores_e, ((0,0), (0,padding)),
                              mode='constant')
        else:
            padding = N2-N
            boxes_g = np.pad(boxes_g, ((0,0), (0,padding), (0,0)),
                             mode='constant')

        overlaps = np.linspace(0.1, 0.9, 9, endpoint=True)

        maps = []
        for overlap in overlaps:
            map3d, precision, recall, overlaps = self.do_eval(boxes_g, boxes_e, scores_e, min_overlap=overlap)
            maps.append(map3d)

        maps = np.stack(maps, axis=0).astype(np.float32)

        return maps

    def do_eval(self, gt_annos, dt_annos, dt_scores, min_overlap=0.7):
        gt_annos = makeDict(gt_annos)
        dt_annos = makeDict(dt_annos)
        for a,s in zip(dt_annos,dt_scores):
            a.update({'score':s})
        # st()
        dt_annos = self.removeInvalidBoxes(dt_annos)
        min_overlaps = np.tile(np.array(min_overlap).reshape(1,1,1), [1,3,1])
        difficultys = [0]
        current_classes = [0]
        mAP_3d, ret, overlaps = self.do_eval_v2(gt_annos, dt_annos, current_classes,
                                           min_overlaps, False, difficultys)
        mAP_3d = mAP_3d.reshape(-1)[0]
        precision, recall = ret['precision'].reshape(-1), ret['recall'].reshape(-1)
        # overlaps = overlaps[0].reshape(-1)
        return mAP_3d, precision, recall, overlaps


    def do_eval_v2(self, gt_annos,
               dt_annos,
               current_classes,
               min_overlaps,
               compute_aos=False,
               difficultys = [0, 1, 2]):

        ret,overlaps = self.eval_class_v3(gt_annos, dt_annos, current_classes, difficultys, 2,
                            min_overlaps)
        mAP_3d = self.get_mAP_v2(ret["precision"])
        # return mAP_bbox, mAP_bev, mAP_3d, mAP_aos
        return mAP_3d, ret, overlaps


    def get_mAP_v2(self, prec):
        sums = 0
        for i in list(range(0, prec.shape[-1], 4)):
            sums = sums + prec[..., i]
        return sums / 11 * 100

    def removeInvalidBoxes(self, dt_annos):
        for frame in dt_annos:
            #frame['score'] = np.ones_like(frame['score'])
            if 'score' in frame:
                valid = frame['score'] > 0
                frame['score'] = frame['score'][valid]
            else:
                valid = np.sum(frame['dimensions'], axis=1) == 0.0
            frame['location'] = frame['location'][valid]
            frame['dimensions'] = frame['dimensions'][valid]
            frame['rotation_y'] = frame['rotation_y'][valid]
        return dt_annos

    def _prepare_data(self, gt_annos, dt_annos, current_class, difficulty):
        gt_datas_list = [np.zeros([f['location'].shape[0],5],
                                  dtype=np.float64)
                         for f in gt_annos]
        dt_datas_list = [np.concatenate([np.zeros([f['location'].shape[0],5],
                                                  dtype=np.float64),
                                         f['score'].reshape(-1,1)],
                                        axis=1)
                         for f in dt_annos]
        ignored_gts = [np.zeros(f['location'].shape[0], dtype=np.int64)
                       for f in gt_annos]
        ignored_dts = [np.zeros(f['location'].shape[0], dtype=np.int64)
                       for f in dt_annos]
        dontcares = [np.zeros([0,4], dtype=np.float64)
                     for f in gt_annos]
        total_dc_num = np.zeros(len(gt_annos), dtype=np.int64)
        total_num_valid_gt = sum(f['location'].shape[0]
                                 for f in gt_annos)
        return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dts,
                dontcares, total_dc_num, total_num_valid_gt)

    def eval_class_v3(self,gt_annos,
                  dt_annos,
                  current_classes,
                  difficultys,
                  metric,
                  min_overlaps,
                  compute_aos=False,
                  num_parts=50):
        """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
        Args:
            gt_annos: dict, must from get_label_annos() in kitti_common.py
            dt_annos: dict, must from get_label_annos() in kitti_common.py
            current_class: int, 0: car, 1: pedestrian, 2: cyclist
            difficulty: int. eval difficulty, 0: easy, 1: normal, 2: hard
            metric: eval type. 0: bbox, 1: bev, 2: 3d
            min_overlap: float, min overlap. official: 
                [[0.7, 0.5, 0.5], [0.7, 0.5, 0.5], [0.7, 0.5, 0.5]] 
                format: [metric, class]. choose one from matrix above.
            num_parts: int. a parameter for fast calculate algorithm

        Returns:
            dict of recall, precision and aos
        """
        assert len(gt_annos) == len(dt_annos)
        num_examples = len(gt_annos)
        split_parts = self.get_split_parts(num_examples, num_parts)

        rets = self.calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
        overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
        N_SAMPLE_PTS = 41
        num_minoverlap = len(min_overlaps)
        num_class = len(current_classes)
        num_difficulty = len(difficultys)
        precision = np.zeros(
            [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
        recall = np.zeros(
            [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
        aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
        for m, current_class in enumerate(current_classes):
            for l, difficulty in enumerate(difficultys):
                rets = self._prepare_data(gt_annos, dt_annos, current_class, difficulty)
                (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
                 dontcares, total_dc_num, total_num_valid_gt) = rets
                for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                    thresholdss = []
                    for i in list(range(len(gt_annos))):
                        rets = self.compute_statistics_jit(
                            overlaps[i],
                            gt_datas_list[i],
                            dt_datas_list[i],
                            ignored_gts[i],
                            ignored_dets[i],
                            dontcares[i],
                            metric,
                            min_overlap=min_overlap,
                            thresh=0.0,
                            compute_fp=False)
                        tp, fp, fn, similarity, thresholds = rets
                        thresholdss += thresholds.tolist()
                    thresholdss = np.array(thresholdss)
                    thresholds = self.get_thresholds(thresholdss, total_num_valid_gt)
                    thresholds = np.array(thresholds)
                    pr = np.zeros([len(thresholds), 4])
                    idx = 0
                    for j, num_part in enumerate(split_parts):
                        gt_datas_part = np.concatenate(
                            gt_datas_list[idx:idx + num_part], 0)
                        dt_datas_part = np.concatenate(
                            dt_datas_list[idx:idx + num_part], 0)
                        dc_datas_part = np.concatenate(
                            dontcares[idx:idx + num_part], 0)
                        ignored_dets_part = np.concatenate(
                            ignored_dets[idx:idx + num_part], 0)
                        ignored_gts_part = np.concatenate(
                            ignored_gts[idx:idx + num_part], 0)
                        self.fused_compute_statistics(
                            parted_overlaps[j],
                            pr,
                            total_gt_num[idx:idx + num_part],
                            total_dt_num[idx:idx + num_part],
                            total_dc_num[idx:idx + num_part],
                            gt_datas_part,
                            dt_datas_part,
                            dc_datas_part,
                            ignored_gts_part,
                            ignored_dets_part,
                            metric,
                            min_overlap=min_overlap,
                            thresholds=thresholds,
                            compute_aos=compute_aos)
                        idx += num_part
                    for i in list(range(len(thresholds))):
                        recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                        precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                        if compute_aos:
                            aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                    for i in list(range(len(thresholds))):
                        precision[m, l, k, i] = np.max(
                            precision[m, l, k, i:], axis=-1)
                        recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
                        if compute_aos:
                            aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
        ret_dict = {
            "recall": recall,
            "precision": precision,
            "orientation": aos,
        }
        return ret_dict, overlaps


    #@numba.jit
    def get_thresholds(self, scores, num_gt, num_sample_pts=41):
        scores.sort()
        scores = scores[::-1]
        current_recall = 0
        thresholds = []
        for i, score in enumerate(scores):
            l_recall = (i + 1) / num_gt
            if i < (len(scores) - 1):
                r_recall = (i + 2) / num_gt
            else:
                r_recall = l_recall
            if (((r_recall - current_recall) < (current_recall - l_recall))
                    and (i < (len(scores) - 1))):
                continue
            # recall = l_recall
            thresholds.append(score)
            current_recall += 1 / (num_sample_pts - 1.0)
        # print(len(thresholds), len(scores), num_gt)
        return thresholds




    #@numba.jit(nopython=True)
    def fused_compute_statistics(self, overlaps,
                                 pr,
                                 gt_nums,
                                 dt_nums,
                                 dc_nums,
                                 gt_datas,
                                 dt_datas,
                                 dontcares,
                                 ignored_gts,
                                 ignored_dets,
                                 metric,
                                 min_overlap,
                                 thresholds,
                                 compute_aos=False):
        gt_num = 0
        dt_num = 0
        dc_num = 0
        for i in list(range(gt_nums.shape[0])):
            for t, thresh in enumerate(thresholds):
                overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:
                                   gt_num + gt_nums[i]]

                gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
                dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
                ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
                ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
                dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
                tp, fp, fn, similarity, _ = self.compute_statistics_jit(
                    overlap,
                    gt_data,
                    dt_data,
                    ignored_gt,
                    ignored_det,
                    dontcare,
                    metric,
                    min_overlap=min_overlap,
                    thresh=thresh,
                    compute_fp=True,
                    compute_aos=compute_aos)
                pr[t, 0] += tp
                pr[t, 1] += fp
                pr[t, 2] += fn
                if similarity != -1:
                    pr[t, 3] += similarity
            gt_num += gt_nums[i]
            dt_num += dt_nums[i]
            dc_num += dc_nums[i]

    #@numba.jit(nopython=True)
    def compute_statistics_jit(self, overlaps,
                               gt_datas,
                               dt_datas,
                               ignored_gt,
                               ignored_det,
                               dc_bboxes,
                               metric,
                               min_overlap,
                               thresh=0,
                               compute_fp=False,
                               compute_aos=False):
        det_size = dt_datas.shape[0]
        gt_size = gt_datas.shape[0]
        dt_scores = dt_datas[:, -1]
        dt_alphas = dt_datas[:, 4]
        gt_alphas = gt_datas[:, 4]
        dt_bboxes = dt_datas[:, :4]
        # gt_bboxes = gt_datas[:, :4]

        assigned_detection = [False] * det_size
        ignored_threshold = [False] * det_size
        if compute_fp:
            for i in list(range(det_size)):
                if (dt_scores[i] < thresh):
                    ignored_threshold[i] = True
        NO_DETECTION = -10000000
        tp, fp, fn, similarity = 0, 0, 0, 0
        # thresholds = [0.0]
        # delta = [0.0]
        thresholds = np.zeros((gt_size, ))
        thresh_idx = 0
        delta = np.zeros((gt_size, ))
        delta_idx = 0
        for i in list(range(gt_size)):
            if ignored_gt[i] == -1:
                continue
            det_idx = -1
            valid_detection = NO_DETECTION
            max_overlap = 0
            assigned_ignored_det = False

            for j in list(range(det_size)):
                if (ignored_det[j] == -1):
                    continue
                if (assigned_detection[j]):
                    continue
                if (ignored_threshold[j]):
                    continue
                overlap = overlaps[j, i]
                dt_score = dt_scores[j]
                if (not compute_fp and (overlap > min_overlap)
                        and dt_score > valid_detection):
                    det_idx = j
                    valid_detection = dt_score
                elif (compute_fp and (overlap > min_overlap)
                      and (overlap > max_overlap or assigned_ignored_det)
                      and ignored_det[j] == 0):
                    max_overlap = overlap
                    det_idx = j
                    valid_detection = 1
                    assigned_ignored_det = False
                elif (compute_fp and (overlap > min_overlap)
                      and (valid_detection == NO_DETECTION)
                      and ignored_det[j] == 1):
                    det_idx = j
                    valid_detection = 1
                    assigned_ignored_det = True

            if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
                fn += 1
            elif ((valid_detection != NO_DETECTION)
                  and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
                assigned_detection[det_idx] = True
            elif valid_detection != NO_DETECTION:
                # only a tp add a threshold.
                tp += 1
                # thresholds.append(dt_scores[det_idx])
                thresholds[thresh_idx] = dt_scores[det_idx]
                thresh_idx += 1
                if compute_aos:
                    # delta.append(gt_alphas[i] - dt_alphas[det_idx])
                    delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                    delta_idx += 1

                assigned_detection[det_idx] = True
        if compute_fp:
            for i in list(range(det_size)):
                if (not (assigned_detection[i] or ignored_det[i] == -1
                         or ignored_det[i] == 1 or ignored_threshold[i])):
                    fp += 1
            nstuff = 0
            if metric == 0:
                overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
                for i in list(range(dc_bboxes.shape[0])):
                    for j in list(range(det_size)):
                        if (assigned_detection[j]):
                            continue
                        if (ignored_det[j] == -1 or ignored_det[j] == 1):
                            continue
                        if (ignored_threshold[j]):
                            continue
                        if overlaps_dt_dc[j, i] > min_overlap:
                            assigned_detection[j] = True
                            nstuff += 1
            fp -= nstuff
            if compute_aos:
                tmp = np.zeros((fp + delta_idx, ))
                # tmp = [0] * fp
                for i in list(range(delta_idx)):
                    tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                    # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
                # assert len(tmp) == fp + tp
                # assert len(delta) == tp
                if tp > 0 or fp > 0:
                    similarity = np.sum(tmp)
                else:
                    similarity = -1
        return tp, fp, fn, similarity, thresholds[:thresh_idx]

    def get_split_parts(self, num, num_part):
        same_part = num // num_part
        remain_num = num % num_part
        if same_part == 0:
            return [remain_num]
        elif remain_num == 0:
            return [same_part] * num_part
        else:
            return [same_part] * num_part + [remain_num]

    def calculate_iou_partly(self, gt_annos, dt_annos, metric, num_parts=50):
        """fast iou algorithm. this function can be used independently to
        do result analysis. Must be used in CAMERA coordinate system.
        Args:
            gt_annos: dict, must from get_label_annos() in kitti_common.py
            dt_annos: dict, must from get_label_annos() in kitti_common.py
            metric: eval type. 0: bbox, 1: bev, 2: 3d
            num_parts: int. a parameter for fast calculate algorithm
        """
        assert len(gt_annos) == len(dt_annos)
        total_dt_num = np.stack([len(a["location"]) for a in dt_annos], 0)
        total_gt_num = np.stack([len(a["location"]) for a in gt_annos], 0)
        num_examples = len(gt_annos)
        split_parts = self.get_split_parts(num_examples, num_parts)
        parted_overlaps = []
        example_idx = 0

        for num_part in split_parts:
            gt_annos_part = gt_annos[example_idx:example_idx + num_part]
            dt_annos_part = dt_annos[example_idx:example_idx + num_part]
            if metric == 0:
                gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
                dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
                overlap_part = self.image_box_overlap(gt_boxes, dt_boxes)
            elif metric == 1:
                loc = np.concatenate(
                    [a["location"][:, [0, 2]] for a in gt_annos_part], 0)
                dims = np.concatenate(
                    [a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
                gt_boxes = np.concatenate(
                    [loc, dims, rots[..., np.newaxis]], axis=1)
                loc = np.concatenate(
                    [a["location"][:, [0, 2]] for a in dt_annos_part], 0)
                dims = np.concatenate(
                    [a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
                dt_boxes = np.concatenate(
                    [loc, dims, rots[..., np.newaxis]], axis=1)
                overlap_part = self.bev_box_overlap(gt_boxes, dt_boxes).astype(
                    np.float64)
            elif metric == 2:
                loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
                dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
                gt_boxes = np.concatenate(
                    [loc, dims, rots[..., np.newaxis]], axis=1)
                loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
                dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
                dt_boxes = np.concatenate(
                    [loc, dims, rots[..., np.newaxis]], axis=1)
                overlap_part = self.d3_box_overlap(gt_boxes, dt_boxes).astype(
                    np.float64)
            else:
                raise ValueError("unknown metric")
            parted_overlaps.append(overlap_part)
            example_idx += num_part
        overlaps = []
        example_idx = 0
        for j, num_part in enumerate(split_parts):
            gt_annos_part = gt_annos[example_idx:example_idx + num_part]
            dt_annos_part = dt_annos[example_idx:example_idx + num_part]
            gt_num_idx, dt_num_idx = 0, 0
            for i in list(range(num_part)):
                gt_box_num = total_gt_num[example_idx + i]
                dt_box_num = total_dt_num[example_idx + i]
                overlaps.append(
                    parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                       dt_num_idx:dt_num_idx + dt_box_num])
                gt_num_idx += gt_box_num
                dt_num_idx += dt_box_num
            example_idx += num_part

        return overlaps, parted_overlaps, total_gt_num, total_dt_num


    def d3_box_overlap_kernel(self, boxes, qboxes, rinc, criterion=-1):
        # ONLY support overlap in CAMERA, not lider.
        N, K = boxes.shape[0], qboxes.shape[0]
        for i in list(range(N)):
            for j in list(range(K)):
                if rinc[i, j] > 0:
                    iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                        boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))

                    if iw > 0:
                        area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                        area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                        inc = iw * rinc[i, j]
                        if criterion == -1:
                            ua = (area1 + area2 - inc)
                        elif criterion == 0:
                            ua = area1
                        elif criterion == 1:
                            ua = area2
                        else:
                            ua = 1.0
                        rinc[i, j] = inc / (self.EPS + ua)
                    else:
                        rinc[i, j] = 0.0

    def d3_box_overlap(self, boxes, qboxes, criterion=-1):
        rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                                   qboxes[:, [0, 2, 3, 5, 6]], 2)
        self.d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
        return rinc

    def bev_box_overlap(self, boxes, qboxes, criterion=-1):
        assert(False) # rotate_iou_gpu_eval (from the old nms_gpu) seems to require cudatoolkit=7.5, which seems unavailable
        # riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
        return riou

    def image_box_overlap(self, boxes, query_boxes, criterion=-1):
        N = boxes.shape[0]
        K = query_boxes.shape[0]
        overlaps = np.zeros((N, K), dtype=boxes.dtype)
        for k in list(range(K)):
            qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                         (query_boxes[k, 3] - query_boxes[k, 1]))
            for n in list(range(N)):
                iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                      max(boxes[n, 0], query_boxes[k, 0]))
                if iw > 0:
                    ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                          max(boxes[n, 1], query_boxes[k, 1]))
                    if ih > 0:
                        if criterion == -1:
                            ua = (
                                (boxes[n, 2] - boxes[n, 0]) *
                                (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                        elif criterion == 0:
                            ua = ((boxes[n, 2] - boxes[n, 0]) *
                                  (boxes[n, 3] - boxes[n, 1]))
                        elif criterion == 1:
                            ua = qbox_area
                        else:
                            ua = 1.0
                        overlaps[n, k] = iw * ih / ua
        return overlaps