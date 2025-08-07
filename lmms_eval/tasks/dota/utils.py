from PIL import Image
import re

COCO_REC_METRICS = ["IoU", "ACC@0.1", "ACC@0.3", "ACC@0.5", "ACC@0.7", "ACC@0.9", "Center_ACC"]

# PROCESSING FUNCTIONS
def dota_filter_grounding(dataset):
    return dataset.filter(lambda x: x["question_type"] == "Grounding")

def dota_filter_classification(dataset):
    return dataset.filter(lambda x: x["question_type"] == "Classification")

def dota_filter_captioning(dataset):
    return dataset.filter(lambda x: x["question_type"] == "Captioning")

def dota_filter_counting(dataset):
    return dataset.filter(lambda x: x["question_type"] == "Counting")

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
    return [0, 0, 0, 0]


def dota_bbox_rec_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = result[0] if len(result) > 0 else ""
    pred = parse_float_sequence_within(pred)
    anno_id = doc["id"]
    data_dict = {"anno_id": anno_id, "pred": pred,  "answer": doc["answer"]}
    return {f"dota_{metric}": data_dict for metric in COCO_REC_METRICS}

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
    Match GT bboxes to pred bboxes using greedy matching.
    Returns list of best scores for each GT box.
    """
    if not gt_bboxes:
        return [0.0] * len(pred_bboxes)
    if not pred_bboxes:
        return [0.0] * len(gt_bboxes)

    matched = set()
    scores = []
    for gt in gt_bboxes:
        best_score = 0.0
        best_pred = None
        for i, pred in enumerate(pred_bboxes):
            if i in matched:
                continue
            score = scorer(gt, pred)
            if score > best_score:
                best_score = score
                best_pred = i
        if best_pred is not None:
            matched.add(best_pred)
        scores.append(best_score)
    return scores

def dota_bbox_rec_aggregation_result(results, metric):
    """
    Aggregate the results of the dota evaluation task using the specified metric.

    Args:
    - results (list of dict): List of result dictionaries.
    - metric (str): Metric to use for aggregation.

    Returns:
    - dict: Dictionary containing the aggregated results for the specified metric.
    """
    
    scorers = {
        "IoU": compute_iou,
        "ACC@0.1": lambda x, y: float(compute_accuracy(x, y, 0.1)),
        "ACC@0.3": lambda x, y: float(compute_accuracy(x, y, 0.3)),
        "ACC@0.5": lambda x, y: float(compute_accuracy(x, y, 0.5)),
        "ACC@0.7": lambda x, y: float(compute_accuracy(x, y, 0.7)),
        "ACC@0.9": lambda x, y: float(compute_accuracy(x, y, 0.9)),
        "Center_ACC": lambda x, y: float(compute_center_accuracy(x, y)),
    }
    
    scorer = scorers[metric]
    scores_all = []
    for result in results:
        gt_bboxes = result["bbox"]
        pred_bboxes = result["pred"]
        scores = match_bboxes(gt_bboxes, pred_bboxes, scorer)
        avg_score = sum(scores) / len(scores) if scores else 0.0
        scores_all.append(avg_score)
    final_score = sum(scores_all) / len(scores_all)
    print(f"Aggregated {metric} score: {final_score}")
    return final_score


def dota_bbox_rec_iou(results):
    return dota_bbox_rec_aggregation_result(results, "IoU")


def dota_bbox_rec_acc01(results):
    return dota_bbox_rec_aggregation_result(results, "ACC@0.1")


def dota_bbox_rec_acc03(results):
    return dota_bbox_rec_aggregation_result(results, "ACC@0.3")


def dota_bbox_rec_acc05(results):
    return dota_bbox_rec_aggregation_result(results, "ACC@0.5")


def dota_bbox_rec_acc07(results):
    return dota_bbox_rec_aggregation_result(results, "ACC@0.7")


def dota_bbox_rec_acc09(results):
    return dota_bbox_rec_aggregation_result(results, "ACC@0.9")


def dota_bbox_rec_center_acc(results):
    return dota_bbox_rec_aggregation_result(results, "Center_ACC")
