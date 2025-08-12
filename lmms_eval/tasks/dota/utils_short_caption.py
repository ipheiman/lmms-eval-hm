from PIL import Image
# from PIL import ImageDraw
from pycocoevalcap.eval import Bleu, Cider, COCOEvalCap, Meteor, Rouge, Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO
from loguru import logger as eval_logger

COCO_METRICS = ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"]  # , "SPICE"]

def dota_filter_captioning(dataset):
    return dataset.filter(lambda x: x["question_type"] == "Captioning")

def dota_doc_to_visual(doc):
    return [Image.open(doc["image"]).convert("RGB")]

def dota_doc_to_text(doc):
    question = doc["question"]
    return question

def dota_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case coco_bleu), value: metric value
    """
    pred = result[0] if len(result) > 0 else ""
    ann_id = doc["id"]
    data_dict = {"answer": doc["answer"], "pred": pred, "ann_id": ann_id}
    return {f"dota_{metric}": data_dict for metric in COCO_METRICS}


def dota_aggregation_result(results, metric):
    # scorers = [(Bleu(4), "Bleu_1"), (Bleu(4), "Bleu_2"), (Bleu(4), "Bleu_3"), (Bleu(4), "Bleu_4"), (Meteor(), "METEOR"), (Rouge(), "ROUGE_L"), (Cider(), "CIDEr")]  # , (Spice(), "SPICE")]
    scorers = [(Bleu(1), "Bleu_1"), (Bleu(2), "Bleu_2"), (Bleu(3), "Bleu_3"), (Bleu(4), "Bleu_4"), (Rouge(), "ROUGE_L"), (Cider(), "CIDEr")]  # , (Spice(), "SPICE")]

    scorers_dict = {s[1]: s for s in scorers}

    stored_results = []
    # In order to make the coco eval tools to successfully create index
    # We need at least two dict in the dataset
    # 'annotation' and 'images'
    # 'annotation' exactly reproduce the original annotation
    # 'images' however only need the image id which is contained in the file name
    dataset = {
        "info": {},
        "annotations": [],
        "images": []
    }
    idx = 0
    ann_id = 0
    for result in results:
        stored_results.append({"image_id": idx, "caption": result["pred"]})
        for s in result["answer"]:
            dataset["annotations"].append({"image_id": idx, "caption": s, "id": ann_id})
            ann_id += 1

        dataset["images"].append({"id": idx})
        idx += 1

    coco = COCO()
    # Manually create index here
    coco.dataset = dataset
    coco.createIndex()

    coco_result = coco.loadRes(stored_results)
    coco_eval = COCOEvalCap(coco, coco_result)

    imgIds = coco_eval.params["image_id"]
    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = coco_eval.coco.imgToAnns[imgId]
        res[imgId] = coco_eval.cocoRes.imgToAnns[imgId]

    eval_logger.info("tokenization...")
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    eval_logger.info(f"Computing {metric} scores...")

    score, scores = scorers_dict[metric][0].compute_score(gts, res)
    # coco_eval.setEval(score, metric)

    # When metric is one of the Bleu, score will be a list
    if type(score) == list:
        n = int(metric.split("_")[-1])
        score = score[n - 1]

    return score


def dota_bleu4(results):
    return dota_aggregation_result(results, "Bleu_4")


def dota_bleu3(results):
    return dota_aggregation_result(results, "Bleu_3")


def dota_bleu2(results):
    return dota_aggregation_result(results, "Bleu_2")


def dota_bleu1(results):
    return dota_aggregation_result(results, "Bleu_1")


def dota_meteor(results):
    return dota_aggregation_result(results, "METEOR")


def dota_rougel(results):
    return dota_aggregation_result(results, "ROUGE_L")


def dota_cider(results):
    return dota_aggregation_result(results, "CIDEr")


def dota_spice(results):
    return dota_aggregation_result(results, "SPICE")
