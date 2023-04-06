import cv2
from PIL import Image, ImageDraw
import torch
import numpy as np

import torch.nn as nn

import clip
import time
from segment_anything import build_sam, SamAutomaticMaskGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]

def get_indices_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]

class RefSAMModel(nn.Module):
    """
        Model built off of SAM and CLIP for performing 
        referring image segmentation.
    """

    def __init__(self, sam_model_path="pretrained/sam_vit_h_4b8939.pth"):
        super().__init__()
        # Load the CLIP model
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        # Load the SAM mask generator model
        self.mask_generator = SamAutomaticMaskGenerator(
            build_sam(checkpoint=sam_model_path).to(device)
        )
    
    def segment_image(self, image, segmentation_mask):
        image_array = np.array(image)
        segmented_image_array = np.zeros_like(image_array)
        segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
        segmented_image = Image.fromarray(segmented_image_array)
        black_image = Image.new("RGB", segmentation_mask.shape, (0, 0, 0))
        transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
        transparency_mask[segmentation_mask] = 255
        transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
        black_image.paste(segmented_image, mask=transparency_mask_image)
        return black_image

    def apply_retrieval(self, elements: list[Image.Image], search_text: str) -> int:
        with torch.no_grad():
            preprocessed_images = [self.preprocess(image).to(device) for image in elements]
            tokenized_text = clip.tokenize([search_text]).to(device)
            stacked_images = torch.stack(preprocessed_images)
            image_features = self.clip_model.encode_image(stacked_images)
            text_features = self.clip_model.encode_text(tokenized_text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            probs = 100.0 * image_features @ text_features.T

            return probs[:, 0].softmax(dim=0)

    def forward(self, image, text, threshold=0.1):
        """
            Do a basic approach of applying the mask generator to the image,
            and then using CLIP to match the text to the image.

            See a similar approach here: https://github.com/maxi-w/CLIP-SAM/
        """
        original_image = image.copy()
        # Apply mask to the input image
        # NOTE: Assumes the input image is RGB
        if len(image.shape) == 4:
            image = image.squeeze(0)
        masks = self.mask_generator.generate(image)
        # Get the CLIP text representation. 
        # Produce a bunch of segmention masks
        bounding_boxes = []

        for mask in masks:
            # image = image.detach().cpu().numpy()
            seg_image = self.segment_image(image, mask["segmentation"])
            box = seg_image.crop(convert_box_xywh_to_xyxy(mask["bbox"]))
            bounding_boxes.append(box)
        # Iterate through the masks, pull out the image sections, get clip representations, and match with text
        scores = self.apply_retrieval(bounding_boxes, text[0])
        indices = get_indices_of_values_above_threshold(scores, threshold)

        final_mask = np.zeros_like(masks[0]["segmentation"])
        # segmentation_masks = []
        for seg_idx in indices:
            # segmentation_mask_image = Image.fromarray(masks[seg_idx]["segmentation"].astype('uint8') * 255)
            final_mask += masks[seg_idx]["segmentation"]
            # segmentation_masks.append(segmentation_mask_image)

        # original_image = Image.open(image_path)
        # Return the mask and a visualized image

        return final_mask, None