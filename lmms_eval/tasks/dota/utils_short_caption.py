from PIL import Image
# from PIL import ImageDraw
from pycocoevalcap.eval import Bleu, Cider, COCOEvalCap, Meteor, Rouge, Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO
from loguru import logger as eval_logger

COCO_METRICS = ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "ROUGE_L", "CIDEr"]

def dota_filter_captioning(dataset):
    return dataset.filter(lambda x: x["question_type"] == "Captioning")

def dota_doc_to_visual(doc):
    return [Image.open(doc["image"]).convert("RGB")]

def dota_doc_to_text(doc):
    question = doc["question"]
    return question

def dota_caption_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case coco_bleu), value: metric value
    """
    pred = result[0] if len(result) > 0 else ""
    answer = doc["answer"]
    anno_id = doc["id"]

    base_data = {"anno_id": anno_id, "pred": pred, "answer": answer}
    output = {}    
    for metric in COCO_METRICS:
        score = dota_captioning_sample_score(base_data, metric) # calculate per-sample score
        # Include score in each metric dict        
        data_with_score = base_data.copy()        
        data_with_score["score"] = score        
        output[f"dota_{metric}"] = data_with_score
    return output



def dota_captioning_sample_score(result, metric):
    """
    Calculate the score for a single caption sample.

    Args:
        result (dict): Must have keys:
            - "pred": predicted caption (str)
            - "answer": list of reference captions (list of str)
        metric (str): One of ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "ROUGE_L", "CIDEr"]

    Returns:
        float: score for this sample
    """
    scorers = {
        "Bleu_1": Bleu(1),
        "Bleu_2": Bleu(2),
        "Bleu_3": Bleu(3),
        "Bleu_4": Bleu(4),
        "ROUGE_L": Rouge(),
        "CIDEr": 0
    }

    scorer = scorers.get(metric)
    if scorer is None:
        raise ValueError(f"Unknown metric: {metric}")

    if isinstance(result["answer"], str):
        answers = [result["answer"]]
    else:
        # list
        answers = result["answer"]


    # Ground truths: list of dicts with "caption" keys
    gts = {0: [{"caption": ans} for ans in answers]}

    # Predictions: must also be a list of dicts
    res = {0: [{"caption": result["pred"]}]}

    # Tokenize like COCOEvalCap does
    tokenizer = PTBTokenizer()
    gts_tok = tokenizer.tokenize(gts)
    res_tok = tokenizer.tokenize(res)

    if metric == "CIDEr":
        return 0
    
    score, _ = scorer.compute_score(gts_tok, res_tok)

    # BLEU returns a list of scores, pick the right n
    if isinstance(score, list):
        n = int(metric.split("_")[-1])
        score = score[n - 1]

    return score

def dota_captioning_aggregation_result(results, metric):
    """
    Aggregate the results of the dota evaluation task using the specified metric.

    Args:
    - results (list of dict): List of result dictionaries.
    - metric (str): Metric to use for aggregation.

    Returns:
    - float: The aggregated score for the specified metric.
    """
    if metric == "CIDEr":
        references = {}
        candidates = {}
        for idx, r in enumerate(results):
            answers = [r["answer"]] if isinstance(r["answer"], str) else r["answer"]
            references[idx] = answers
            candidates[idx] = [r["pred"]]

        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(references, candidates)
        return cider_score

    scores_all = []
    for result in results:
        scores_all.append(result["score"])
    final_score = sum(scores_all) / len(scores_all)
    return final_score


def dota_bleu4(results):
    return dota_captioning_aggregation_result(results, "Bleu_4")


def dota_bleu3(results):
    return dota_captioning_aggregation_result(results, "Bleu_3")


def dota_bleu2(results):
    return dota_captioning_aggregation_result(results, "Bleu_2")


def dota_bleu1(results):
    return dota_captioning_aggregation_result(results, "Bleu_1")


def dota_rougel(results):
    return dota_captioning_aggregation_result(results, "ROUGE_L")


def dota_cider(results):
    return dota_captioning_aggregation_result(results, "CIDEr")

