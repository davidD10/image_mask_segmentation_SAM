import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

image_path = "images/highway_image.jpg"  
image = Image.open(image_path)
image = np.array(image)

# Display the original image
plt.imshow(image)
plt.axis('off')
plt.title("Original Image")
plt.show()

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)
print(f"Number of masks generated: {len(masks)}")

def random_color():
    return [random.randint(0, 255) for _ in range(3)]

rgb_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

for mask in masks:
    color = random_color()  
    mask_indices = mask['segmentation']  
    rgb_mask[mask_indices] = color  

# Display the RGB mask with each segment in a different color
plt.imshow(rgb_mask)
plt.axis('off')
plt.title("Segmented Masks")
plt.show()

alpha = 0.5  
overlay = cv2.addWeighted(image, 1 - alpha, rgb_mask, alpha, 0)

# Display the combined result
plt.imshow(overlay)
plt.axis('off')
plt.title("Overlay of Segmentation Masks on Original Image")
plt.show()

result_image = Image.fromarray(overlay)
result_image.save("segmented_highway_cars.png")
print("Saved the segmented image as 'segmented_highway_cars.png'")