import math
from collections import defaultdict
import os
import argparse
from tqdm import tqdm
import json

def parse_annotation(line):
    parts = line.strip().split()
    coords = list(map(float, [parts[0], parts[1], parts[4], parts[5]]))
    # ignore difficulty 0/1
    cls = parts[8]
    return coords, cls

def format_bbox(coords):
    return "[" + ", ".join(str(int(c)) for c in coords) + "]"

def generate_vqa(annotations):
    '''
    Generates questions PER .txt file
    Generates 1 existence question
    Generates 1 counting question
    Generates 2 Localization (Box → Class) questions
    Generates 3 Localization (Class → Box) questions
    Total: 7 questions per annotation
    '''
    class_counts = defaultdict(int)
    class_to_bboxes = defaultdict(list)

    for coords, cls in annotations:
        class_counts[cls] += 1
        class_to_bboxes[cls].append(coords)    

    vqa_data = []

    # 1. Existence Questions
    for cls in class_counts:
        q = f"Is there a {cls.replace('-', ' ')} in the image? Answer the question with Yes or No."
        a = "yes"
        vqa_data.append({"question": q, "answer": a})

    # 2. Counting Questions
    for cls, count in class_counts.items():
        q = f"How many {cls.replace('-', ' ')}s are there in the image? Answer the question with a single word."
        a = str(count)
        vqa_data.append({"question": q, "answer": a})

    # 3. Localization (Box → Class)
    for coords, cls in annotations:
        bbox_str = format_bbox(coords)
        
        # 2 Templates
        q1 = f"What is the object in the bounding box {bbox_str}? Answer the question with a single word (or phrase)."
        q2 = f"Given the bounding box {bbox_str}, what object is shown? Answer the question with a single word (or phrase)."

        a = cls.replace('-', ' ')

        vqa_data.append({"question": q1, "answer": a})
        vqa_data.append({"question": q2, "answer": a})

    # 4. Localization (Class → Box)
    for cls, bboxes in class_to_bboxes.items():
        if not bboxes:
            continue

        # 3 Templates
        q1 = f"What are the bounding box coordinates of the {cls.replace('-', ' ')} in the image? Provide the bounding box coordinates in the form of [x1, y1, x2, y2]."
        q2 = f"Give the coordinates of the {cls.replace('-', ' ')}. Provide the bounding box coordinates in the form of [x1, y1, x2, y2]."
        q3 = f"Where is the {cls.replace('-', ' ')} in this image? Provide the bounding box coordinates in the form of [x1, y1, x2, y2]."

        answer_list = [format_bbox(coords) for coords in bboxes]
        a = "; ".join(answer_list)

        vqa_data.append({"question": q1, "answer": a})
        vqa_data.append({"question": q2, "answer": a})
        vqa_data.append({"question": q3, "answer": a})                

    return vqa_data


def main(input_folder, output_file):
    annotation_files = os.listdir(input_folder)

    # Extract annotations
    for file in tqdm(annotation_files):
        annotation_file_path = os.path.join(input_folder,file)
        file_name = os.path.splitext(file)[0]
        all_annotations = []

        with open(annotation_file_path, "r") as f:
            # skip first 2 lines about imagesource and gsd         
            lines = f.readlines()[2:] 
            for line in lines:
                coords, cls = parse_annotation(line)
                # list of dictionary for a single image
                all_annotations.append((coords, cls))
            
            # generate vqa for this image's annotations
            vqa_pairs = generate_vqa(all_annotations)

            with open(f"{output_file}.jsonl", "a") as out_f:
                for pair in vqa_pairs:
                    pair_with_file = {
                        "image": file_name + ".png",
                        "question": pair["question"],
                        "answer": pair["answer"]
                    }                    

                    json.dump(pair_with_file, out_f)
                    out_f.write("\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate VQA pairs from annotations")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the annotation directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Name of output .jsonl file")
    args = parser.parse_args()

    main(args.input, args.output)

