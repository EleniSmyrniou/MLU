import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import torch
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
torch.cuda.empty_cache()
# Set cuBLAS to CPU fallback mode
import os
os.environ['CUBLAS_FALLBACK_MODE'] = '1'


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


if __name__=="__main__":
    image = cv2.imread('D:/MLU/test_images/test_3.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis('off')

    sam_checkpoint = "D:/MLU/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    #sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=80,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    torch.cuda.empty_cache()
    masks = mask_generator.generate(image)

    print(len(masks))
    print(masks[0].keys())

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()






