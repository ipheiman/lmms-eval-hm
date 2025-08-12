from PIL import Image

def dota_filter_classification(dataset):
    return dataset.filter(lambda x: x["question_type"] == "Classification")


def dota_doc_to_visual(doc):
    return [Image.open(doc["image"]).convert("RGB")]

def dota_doc_to_text(doc):
    question = doc["question"]
    return question

# Draw bounding box on img, before passing to model
# Should we do this?
# def dota_bbox_doc_to_visual(doc):
#     bbox = doc["bbox"]
#     image = doc["image"].convert("RGB")
#     draw = ImageDraw.Draw(image)
#     # Origin format (top x, top y, width, height)
#     bbox_xy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
#     draw.rectangle(bbox_xy, outline="red")
#     return [image.convert("RGB")]