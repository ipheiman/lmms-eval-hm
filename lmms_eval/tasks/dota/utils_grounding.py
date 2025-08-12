from PIL import Image
import re
from scipy.optimize import linear_sum_assignment
import numpy as np

COCO_REC_METRICS = ["IoU", "ACC@0.1", "ACC@0.3", "ACC@0.5", "ACC@0.7", "ACC@0.9", "Center_ACC"]

# PROCESSING FUNCTIONS
def dota_filter_grounding(dataset):
    return dataset.filter(lambda x: x["question_type"] == "Grounding")

def dota_doc_to_visual(doc):
    return [Image.open(doc["image"]).convert("RGB")]

def dota_doc_to_text(doc):
    question = doc["question"]
    return question


# GROUNDING FUNCTIONS
def parse_float_sequence_within(input_str):
    """
    Extract all sequences of four floating-point numbers within square brackets from a string.

    Args:
    input_str (str): A string that may contain multiple sequences of four floats within square brackets.

    Returns:
    list: A list of bounding boxes, where each bounding box is a list of four floats.
          Returns an empty list if no valid bounding boxes are found.
    """
    # Define the regex pattern to find all instances of four floats within square brackets
    pattern = r"\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]"

    # Use re.findall to find all matches of the pattern in the input string
    matches = re.findall(pattern, input_str)

    # If matches are found, convert each group of matches into a list of floats
    if matches:
        return [[float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in matches]

    # If the input does not contain any valid bounding boxes, return an empty list
    return [[0, 0, 0, 0]]


def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - float: IoU of box1 and box2.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Compute the area of intersection
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the area of the union
    union_area = box1_area + box2_area - intersection_area

    # Compute the Intersection over Union
    iou = intersection_area / union_area

    return iou


def compute_accuracy(box1, box2, threshold=0.5):
    """
    Compute the accuracy of two bounding boxes based on a specified threshold.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - threshold (float): Threshold for the IoU to consider the prediction correct.

    Returns:
    - float: Accuracy of the prediction based on the IoU threshold.
    """
    iou = compute_iou(box1, box2)
    return iou >= threshold


def compute_center_accuracy(box1, box2):
    """
    Compute if the center point of box 2 is within box 1.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - bool: True if the center point of box 2 is within box 1, False otherwise.
    """
    # Compute the center point of box 2
    center_x = (box2[0] + box2[2]) / 2
    center_y = (box2[1] + box2[3]) / 2

    # Check if the center point is within box 1
    return box1[0] <= center_x <= box1[2] and box1[1] <= center_y <= box1[3]

def match_bboxes(gt_bboxes, pred_bboxes, scorer):
    """
    Match GT bboxes to pred bboxes using Hungarian matching.
    Returns list of best scores for each GT box (0 if no match).

    Args:
        gt_bboxes (list): List of ground truth boxes.
        pred_bboxes (list): List of predicted boxes.
        scorer (function): Function to score (gt_box, pred_box).

    Returns:
        List[float]: Best scores for each GT box.
    """
    if len(gt_bboxes) == 0:
        # No GT: no scores for GT
        return []
    if len(pred_bboxes) == 0:
        # No predictions: all GT have zero score
        return [0.0] * len(gt_bboxes)

    # Build cost matrix (GT x Pred) by scoring each pair
    cost_matrix = np.zeros((len(gt_bboxes), len(pred_bboxes)), dtype=np.float32)
    for i, gt in enumerate(gt_bboxes):
        for j, pred in enumerate(pred_bboxes):
            score = scorer(gt, pred)
            # We want to maximize score, but linear_sum_assignment minimizes cost,
            # so use negative score as cost.
            cost_matrix[i, j] = -score

    # Hungarian algorithm to find optimal assignment
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    # Prepare results: for each GT box, assign best score or 0 if no assignment
    scores = [0.0] * len(gt_bboxes)
    assigned_preds = set()
    for gt_i, pred_i in zip(gt_indices, pred_indices):
        # Only assign if score > 0 (optional: you can change threshold here)
        if -cost_matrix[gt_i, pred_i] > 0:
            scores[gt_i] = -cost_matrix[gt_i, pred_i]
            assigned_preds.add(pred_i)
        else:
            # Score too low, consider no match
            scores[gt_i] = 0.0

    return scores


def dota_bbox_rec_process_result(doc, result):
    pred = result[0] if len(result) > 0 else ""
    pred = parse_float_sequence_within(pred)
    answer = parse_float_sequence_within(doc["answer"])
    anno_id = doc["id"]

    base_data = {"anno_id": anno_id, "pred": pred, "answer": answer}

    output = {}
    for metric in COCO_REC_METRICS:
        score = dota_bbox_rec_sample_score(base_data, metric)  # calculate per-sample score
        # Include score in each metric dict
        data_with_score = base_data.copy()
        data_with_score["score"] = score
        output[f"dota_{metric}"] = data_with_score

    return output

def dota_bbox_rec_sample_score(result, metric):
    """
    Calculate the score for a single sample based on the given metric.

    Args:
        result (dict): A dictionary containing 'pred' and 'answer' keys with bounding boxes.
        metric (str): Metric name, one of ["IoU", "ACC@0.1", "ACC@0.3", "ACC@0.5", "ACC@0.7", "ACC@0.9", "Center_ACC"].

    Returns:
        float: The computed score for the sample.
    """
    scorers = {
        "IoU": compute_iou,
        "ACC@0.1": lambda gt, pred: compute_accuracy(gt, pred, 0.1),
        "ACC@0.3": lambda gt, pred: compute_accuracy(gt, pred, 0.3),
        "ACC@0.5": lambda gt, pred: compute_accuracy(gt, pred, 0.5),
        "ACC@0.7": lambda gt, pred: compute_accuracy(gt, pred, 0.7),
        "ACC@0.9": lambda gt, pred: compute_accuracy(gt, pred, 0.9),
        "Center_ACC": compute_center_accuracy,
    }

    scorer = scorers.get(metric)
    if scorer is None:
        raise ValueError(f"Unknown metric: {metric}")

    gt_bboxes = result["answer"]
    pred_bboxes = result["pred"]

    # match_bboxes will match predicted to gt and compute per-match scores
    scores = match_bboxes(gt_bboxes, pred_bboxes, scorer)

    # average score over all matches, or 0 if no matches
    return sum(scores) / len(scores) if scores else 0.0

# Calculates based on metric

def dota_bbox_rec_aggregation_result(results):
    """
    Aggregate the results of the dota evaluation task using the specified metric.

    Args:
    - results (list of dict): List of result dictionaries.
    - metric (str): Metric to use for aggregation.

    Returns:
    - dict: Dictionary containing the aggregated results for the specified metric.
    """

    scores_all = []
    for result in results:
        scores_all.append(result["score"])
    final_score = sum(scores_all) / len(scores_all)
    return final_score

# def dota_bbox_rec_aggregation_result(results, metric):
#     """
#     Aggregate the results of the dota evaluation task using the specified metric.

#     Args:
#     - results (list of dict): List of result dictionaries.
#     - metric (str): Metric to use for aggregation.

#     Returns:
#     - dict: Dictionary containing the aggregated results for the specified metric.
#     """

#     print(results)
#     scorers = {
#         "IoU": compute_iou,
#         "ACC@0.1": lambda x, y: compute_accuracy(x, y, 0.1),
#         "ACC@0.3": lambda x, y: compute_accuracy(x, y, 0.3),
#         "ACC@0.5": lambda x, y: compute_accuracy(x, y, 0.5),
#         "ACC@0.7": lambda x, y: compute_accuracy(x, y, 0.7),
#         "ACC@0.9": lambda x, y: compute_accuracy(x, y, 0.9),
#         "Center_ACC": lambda x, y: compute_center_accuracy(x, y),
#     }

#     scorer = scorers[metric]
#     scores_all = []
#     for result in results:
#         gt_bboxes = result["answer"]
#         pred_bboxes = result["pred"]
#         scores = match_bboxes(gt_bboxes, pred_bboxes, scorer)
#         avg_score = sum(scores) / len(scores) if scores else 0.0
#         scores_all.append(avg_score)
#     final_score = sum(scores_all) / len(scores_all)
#     print(f"Aggregated {metric} score: {final_score}")
#     return final_score


# def dota_bbox_rec_iou(results):
#     # return dota_bbox_rec_aggregation_result(results, "IoU")
#     return dota_bbox_rec_aggregation_result(results)    


# def dota_bbox_rec_acc01(results):
#     # return dota_bbox_rec_aggregation_result(results, "ACC@0.1")
#     return dota_bbox_rec_aggregation_result(results)    


# def dota_bbox_rec_acc03(results):
#     # return dota_bbox_rec_aggregation_result(results, "ACC@0.3")
#     return dota_bbox_rec_aggregation_result(results)    


# def dota_bbox_rec_acc05(results):
#     # return dota_bbox_rec_aggregation_result(results, "ACC@0.5")
#     return dota_bbox_rec_aggregation_result(results)    


# def dota_bbox_rec_acc07(results):
#     # return dota_bbox_rec_aggregation_result(results, "ACC@0.7")
#     return dota_bbox_rec_aggregation_result(results)    


# def dota_bbox_rec_acc09(results):
#     # return dota_bbox_rec_aggregation_result(results, "ACC@0.1")
#     return dota_bbox_rec_aggregation_result(results)    


# def dota_bbox_rec_center_acc(results):
#     # return dota_bbox_rec_aggregation_result(results, "Center_ACC")
#     return dota_bbox_rec_aggregation_result(results)    

def dota_bbox_rec(results):
    return dota_bbox_rec_aggregation_result(results)    