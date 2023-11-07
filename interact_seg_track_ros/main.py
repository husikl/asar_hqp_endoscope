import sys
import torch
from PIL import Image
import numpy as np
import cv2

from models.XMem.definition import XMem
from models.XMem.inference.inference_core import InferenceCore
from segment_anything import SamPredictor, sam_model_registry

# ! load configs
from configs.XMem import _get_config
from configs.SAM import sam_config

# ! load interaction control
from control.interaction import interaction_control

# ! load resource manager
from control.res_manager import ResManager as resource_manager
import math


# def cal_mass_center(mask):
#     # centeroid of a mask can be obtained by M_{10}/M_{00} and M_{01}/M_{00} from moments
#     # https://en.wikipedia.org/wiki/Image_moment
#     # assert isinstance(mask, np.ndarray)  # n_obj (includes bg), H, W
#     n_obj = mask.shape[0]
#     mask_cal_idv = dict()
#     for i_obj in range(1, n_obj):
#         _, thresh = cv2.threshold(mask[i_obj], 0.5, 1, cv2.THRESH_BINARY)

#         # Calculate the moments to get the centroid
#         M = cv2.moments(thresh)
#         if M["m00"] != 0:
#             cX = int(M["m10"] / M["m00"])
#             cY = int(M["m01"] / M["m00"])
#         else:
#             cX, cY = 0, 0
#         radius = math.sqrt(M["m00"] / math.pi)
#         mask_cal_idv[f"{i_obj+1}"] = (cX, cY, int(radius))
#     # mask_cal = np.sum(mask[1:], axis=0)  # remove bg
#     # m = cv2.moments(mask_cal, False)

#     # x, y = m["m10"] / m["m00"], m["m01"] / m["m00"]
#     return mask_cal_idv
def cal_mass_center(mask):
    n_obj = mask.shape[0]
    mask_cal_idv = {}
    image_height, image_width = mask[0].shape[:2]
    image_center = np.array([image_width / 2, image_height / 2])
    for idx in range(1, n_obj):
        single_mask = mask[idx]
        
        _, thresh = cv2.threshold(single_mask, 0.75, 1, cv2.THRESH_BINARY)

        # Convert to a compatible type if necessary
        if thresh.dtype != np.uint8:
            thresh = (thresh * 255).astype(np.uint8)

        # Apply morphological operations
        kernel = np.ones((3,3), np.uint8)  # Kernel size can be adjusted
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # Erosion followed by dilation (removes noise)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # Dilation followed by erosion (closes small holes)

        # Now proceed with contour extraction
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate mass center
        M = cv2.moments(thresh)
        m00 = M["m00"]
        
        if m00 != 0:
            m10, m01 = M["m10"], M["m01"]
            cX, cY = int(m10 / m00), int(m01 / m00)
            centroid_array = np.array([cX, cY])
            radius = int(math.sqrt(m00 / math.pi))
            
            # Filter contours that include the centroid
            valid_contours = [cnt for cnt in contours if cv2.pointPolygonTest(cnt, (cX, cY), False) >= 0]
            
            # if valid_contours:
            max_contour = max(contours, key=cv2.contourArea)
            # hull = cv2.convexHull(max_contour)
            rect = cv2.minAreaRect(max_contour)
            box = np.int0(cv2.boxPoints(rect))
            max_dist, pt1, pt2 = 0, None, None
            for i in range(4):
                for j in range(i+1, 4):
                    dist = np.linalg.norm(box[i] - box[j])
                    if dist > max_dist:
                        max_dist, pt1, pt2 = dist, box[i], box[j]
            
            
            tip = pt1 if np.linalg.norm(image_center - pt1) < np.linalg.norm(image_center - pt2) else pt2
            # farthest_point = max(hull, key=lambda x: cv2.norm(np.array([cX, cY]) - x[0]))

            # tip = farthest_point[0]
            # mask_cal_idv[f"{idx+1}"] = (tip[0], tip[1], 10, cX, cY)
            mask_cal_idv[f"{idx+1}"] = (cX, cY, 10, tip[0], tip[1])
            # mask_cal_idv[f"{idx+1}"] = (cX, cY, 10,  cX, cY)
            # else:
            #     mask_cal_idv[f"{idx+1}"] = (0, 0, 0, 0, 0)
        else:
            mask_cal_idv[f"{idx+1}"] = (0, 0, 0, 0, 0)
            
    return mask_cal_idv


def build_control():
    XMem_config = _get_config()
    network = XMem(XMem_config, XMem_config["model"]).cuda().eval()
    infer_control = InferenceCore(network=network, config=XMem_config)
    res_manager = resource_manager(infer_control)
    sam = SamPredictor(sam_model_registry[sam_config[0]](sam_config[1]).cuda())
    interact_control = interaction_control(sam)
    return res_manager, interact_control


# def init_interactive_segmentation(image, res_manager, interact_control, tk_root=None):
#     assert isinstance(image, np.ndarray)
#     assert image.shape[2] == 3
#     assert len(image.shape) == 3
#     # masks = interact_control.do_interact(Image.fromarray(image)) # box_num, 1 , H, W, prob ranged from 0 to 1
#     masks = interact_control.do_interact(Image.fromarray(image), tk_root)
#     print("got masks from sam...")
#     # print(f'sam masks shape: {masks.shape}')
#     # import pdb; pdb.set_trace()
#     # cv2.imwrite('sam_masks.png', masks[0,0].cpu().numpy()*255)
#     res_manager.set_image(image)
#     res_manager.set_mask(masks)
#     masks = res_manager.step(mask=True)
#     # cal_mass_center(masks.cpu().numpy())
#     return masks
def init_interactive_segmentation(image, res_manager, interact_control, tk_root=None, debug=False):
    if debug:
        assert isinstance(image, np.ndarray)
        assert image.shape[2] == 3
        assert len(image.shape) == 3

    masks = interact_control.do_interact(Image.fromarray(image), tk_root)
    
    if debug:
        print("got masks from sam...")

    res_manager.set_image(image)
    res_manager.set_mask(masks)
    masks = res_manager.step(mask=True)
    
    return masks


# def inference_masks(image, res_manager):
#     assert isinstance(image, np.ndarray)
#     assert image.shape[2] == 3
#     assert len(image.shape) == 3
#     res_manager.set_image(image)
#     masks = res_manager.step(mask=None)
#     send_masks = masks.cpu().numpy()
#     circle = cal_mass_center(send_masks)
#     # center = cal_mass_center(masks.cpu().numpy())
#     # print(center)
#     #  [mask1_x, mask1_y, mask1_label, mask1_min_radius]
#     # masks = {}
#     # masks['label'] = position
#     return circle
def inference_masks(image, res_manager, debug=False):
    if debug:
        assert isinstance(image, np.ndarray)
        assert image.shape[2] == 3
        assert len(image.shape) == 3
    
    res_manager.set_image(image)
    masks = res_manager.step(mask=None)
    
    # Only convert to CPU and numpy array once
    send_masks = masks.cpu().numpy()
    
    circle = cal_mass_center(send_masks)
    
    return circle


if __name__ == "__main__":
    XMem_config = _get_config()
    with torch.cuda.amp.autocast(enabled=True):
        network = XMem(XMem_config, XMem_config["model"]).cuda().eval()
        infer_control = InferenceCore(network=network, config=XMem_config)
        res_manager = resource_manager(infer_control)
        sam = SamPredictor(sam_model_registry[sam_config[0]](sam_config[1]).cuda())
        interact_control = interaction_control(sam, image=Image.open("4050.jpg"))
        masks = (
            interact_control.do_interact()
        )  # box_num, 1 , H, W, prob ranged from 0 to 1
        # print('first')
        # masks = control.do_interact(image=Image.open("4125.jpg"))
        # print('second')
        startnum = 4050
        res_manager.set_image(
            np.array(Image.open(f"{startnum}.jpg"))
        )  # np array H, W, 3
        res_manager.set_mask(masks)
        masks_tmp = res_manager.step(mask=True)
        import cv2
        import time

        t0 = time.time()
        for i in range(5):
            res_manager.set_image(np.array(Image.open(f"{startnum+i*25}.jpg")))
            masks_tmp = res_manager.step(mask=None)
            cv2.imwrite(f"mask_{startnum+i*25}.png", masks_tmp[-1].cpu().numpy() * 255)

        t1 = time.time()
        print("finished, each image cost:", (t1 - t0) / 5)
