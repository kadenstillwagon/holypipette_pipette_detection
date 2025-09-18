import torch
import numpy as np
import cv2
from torchvision import transforms
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from dinov2.dinov2.data.transforms import make_normalize_transform

def get_pipeline(model, device):
    return DINODistanceMapSlidingWindowPipeline(model, device)

class DINODistanceMapSlidingWindowPipeline:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.img_size = 518

    def _preprocess(self, img):
        #adaptive hist normalization
        img_norm = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8)).apply(img)
        img = cv2.normalize(img_norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        #cvt to 3 channel
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        #default preprocessing
        image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
                transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            ]
        )

        img = image_transform(img).unsqueeze(0)

        return img.to(self.device)

    def get_model_prediction(self, image):
        image_orig = image.copy()
        image = self._preprocess(image_orig)
        self.model.eval().to(self.device)

        # forward pass
        with torch.no_grad():
            preds = self.model(x=image)

        xy_pred = preds[0]
        z_coord_pred = preds[1]

        return xy_pred, z_coord_pred

    
    def run(self, image):
        img_orig_shape = image.shape

        xy_pred, z_coord_pred = self.get_model_prediction(image)

        xy_coord_pred = xy_pred[0]

        x_pred = int(xy_coord_pred[0]  + img_orig_shape[1] // 2)
        y_pred = int(xy_coord_pred[1]  + img_orig_shape[0] // 2)

        tip_coord_pred = (x_pred, y_pred, z_coord_pred)

        return tip_coord_pred

