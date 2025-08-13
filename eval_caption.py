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


cider_scorer = Cider()

score, scores = cider_scorer.compute_score(references, candidates)

print(f"Average CIDEr score: {score:.4f}")
# print("CIDEr scores per image:")
# print(scores)