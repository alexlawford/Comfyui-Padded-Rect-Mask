#  Package Modules
import os
import torch
from typing import Union, BinaryIO, Dict, List, Tuple, Optional
import time

#  ComfyUI Modules
import folder_paths
from comfy.utils import ProgressBar

#  Basic practice to get paths from ComfyUI
custom_nodes_script_dir = os.path.dirname(os.path.abspath(__file__))
custom_nodes_model_dir = os.path.join(folder_paths.models_dir, "my-custom-nodes")
custom_nodes_output_dir = os.path.join(folder_paths.get_output_directory(), "my-custom-nodes")

def tensor2rgba(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 4)
    elif size[3] == 1:
        return t.repeat(1, 1, 1, 4)
    elif size[3] == 3:
        alpha_tensor = torch.ones((size[0], size[1], size[2], 1))
        return torch.cat((t, alpha_tensor), dim=3)
    else:
        return t

def convert(image):
    return (tensor2rgba(image)[:,:,:,0],)

#  Rect Mask
class RectMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "x": ("INT", {"default": 0}),
                "y": ("INT", {"default": 0}),
                "width": ("INT", {"default": 0}),
                "height": ("INT", {"default": 0}),
                "image_width": ("INT", {"default": 512}),
                "image_height": ("INT", {"default": 512}),
                "padding": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK")
    FUNCTION = "create_mask"
    CATEGORY = "CustomNodesTemplate"

    def create_mask(self, x, y, width, height, image_width, image_height, padding):
        min_x = x - padding if x - padding > 0 else 0
        min_y = y - padding if y - padding > 0 else 0

        max_x = min_x + width + padding
        max_y = min_y + height + padding
            
        mask = torch.zeros((image_height, image_width))
        mask[int(min_y):int(max_y)+1, int(min_x):int(max_x)+1] = 1
        return convert((mask.unsqueeze(0),))
