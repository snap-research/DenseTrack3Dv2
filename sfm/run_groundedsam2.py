import os
import cv2
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import argparse

BASE_DIR = os.getcwd()
os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "Grounded-SAM-2")))

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

"""
Hyperparam for Ground and Tracking
"""
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.5
TEXT_PROMPT = "a woman"
PROMPT_TYPE_FOR_VIDEO = "box" # choose from ["point", "box", "mask"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# SOURCE = "./vis_results/custom_data/"
# NAME = "lucia"
# SOURCE_VIDEO_FRAME_DIR = os.path.join(SOURCE, NAME, "color")
# SAVE_TRACKING_RESULTS_DIR = os.path.join(SOURCE, NAME, "mask")

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="demo_data/rollerblade", help="demo video path")
    parser.add_argument("--output_path", type=str, default="results/demo", help="output path")
    parser.add_argument(
        "--prompt",  type=str, default="a yellow duck", help="text prompt to segment foreground object"
    )

    return parser

if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    source_video_frame_dir = os.path.join(args.video_path, "color")
    save_videomask_dir = os.path.join(args.output_path, "mask")
    os.makedirs(save_videomask_dir, exist_ok=True)

    """
    Step 1: Environment settings and model initialization for Grounding DINO and SAM 2
    """
    # build grounding dino model from local path
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG, 
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE
    )


    # init sam image predictor and video predictor model
    sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)


    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(source_video_frame_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", "png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # init video predictor state
    inference_state = video_predictor.init_state(video_path=source_video_frame_dir)

    ann_frame_idx = 0  # the frame index we interact with
    """
    Step 2: Prompt Grounding DINO 1.5 with Cloud API for box coordinates
    """

    # prompt grounding dino to get the box coordinates on specific frame
    img_path = os.path.join(source_video_frame_dir, frame_names[ann_frame_idx])
    image_source, image = load_image(img_path)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    confidences = confidences.numpy().tolist()
    class_names = labels

    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(image_source)

    # process the detection results
    OBJECTS = class_names


    # FIXME: figure how does this influence the G-DINO model
    torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # prompt SAM 2 image predictor to get the mask for the object
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    # convert the mask shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    """
    Step 3: Register each object's positive points to video predictor with seperate add_new_points call
    """

    assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

    # If you are using point prompts, we uniformly sample positive points based on the mask
    if PROMPT_TYPE_FOR_VIDEO == "point":
        # sample the positive points from mask for each objects
        all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

        for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
            labels = np.ones((points.shape[0]), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels,
            )
    # Using box prompt
    elif PROMPT_TYPE_FOR_VIDEO == "box":
        for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                box=box,
            )
    # Using mask prompt is a more straightforward way
    elif PROMPT_TYPE_FOR_VIDEO == "mask":
        for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
            labels = np.ones((1), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                mask=mask
            )
    else:
        raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")


    """
    Step 4: Propagate the video predictor to get the segmentation results for each frame
    """
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    """
    Step 5: Visualize the segment results across the video and save them
    """


    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}

    for frame_idx, segments in video_segments.items():
        img = cv2.imread(os.path.join(save_videomask_dir, frame_names[frame_idx]))
        
        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)

        binary_mask = masks.sum(0) # H W
        binary_mask = binary_mask.astype(np.uint8) * 255


        cv2.imwrite(os.path.join(save_videomask_dir, f"{frame_idx:05d}.png"), binary_mask)
