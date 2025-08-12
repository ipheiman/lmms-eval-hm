from pycocoevalcap.cider.cider import Cider

references = {
            'img1': ['This image shows a satellite view of an airport with multiple terminals and parked airplanes.'],
            'img2': ['This image shows a satellite view of a lakeside residential area with docks extending into the water.'],
            'img3': ['This image shows a satellite view of a park with baseball and soccer fields, parking areas, and surrounding greenery.']
}

candidates = {
    'img1': ['This image is an aerial view of an airport terminal and surrounding area. The terminal is designed with multiple concourses and gates, each equipped with jet bridges for boarding and disembarking passengers. Several airplanes are parked at the gates, indicating active operations.'], 
    'img2': ['The image is an aerial view of a lake surrounded by residential properties. The lake has a dark, reflective surface, and the surrounding area is densely forested with green trees. There are several docks and piers extending into the water from the shore.'],
    'img3': ['The image is an aerial view of a park or recreational area featuring several sports fields and facilities. Here are the key elements visible in the image:\n\n1. **Sports Fields**:\n   - There are multiple grassy fields, likely used for various sports activities.\n2. **Parking Areas**:\n   - Ample parking is available for visitors.\n3. **Surrounding Greenery**:\n   - The park is bordered by trees and natural vegetation, providing a scenic backdrop.'],
}


# Ensure the candidate dictionary has a single generated caption per image ID for evaluation.
# If you have multiple generated captions for an image, select one for CIDEr calculation.
# For simplicity in this example, we'll assume a single candidate per image.
# If 'img1' had two candidates, you'd choose one, e.g., candidates_for_eval = {'img1': 'A cat on a couch.', 'img2': 'People walking on the street.'}

# Create a Cider object.
# n=4 indicates using up to 4-grams for similarity calculation.
# sigma=6.0 is a parameter for the exponential weighting of n-grams.
cider_scorer = Cider()

# Compute the CIDEr score.
# The compute_score method takes two dictionaries:
# 1. The references dictionary (image_id: [list of reference captions])
# 2. The candidates dictionary (image_id: [list of candidate captions - typically one per image])
# It returns the average CIDEr score for the corpus and a dictionary of scores per image.
score, scores = cider_scorer.compute_score(references, candidates)

print(f"Average CIDEr score: {score:.4f}")
# print("CIDEr scores per image:")
# print(scores)