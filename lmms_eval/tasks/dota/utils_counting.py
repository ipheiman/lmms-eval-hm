from PIL import Image

def dota_filter_counting(dataset):
    return dataset.filter(lambda x: x["question_type"] == "Counting")

def dota_doc_to_visual(doc):
    return [Image.open(doc["image"]).convert("RGB")]

def dota_doc_to_text(doc):
    question = doc["question"]
    return question