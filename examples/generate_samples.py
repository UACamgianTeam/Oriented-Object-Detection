"""
Usage: python examples/generate_samples.py ./dota_sports_data
"""

import sys
import os
from PIL import Image
from ood.preprocess import * 
from ood.utils import get_annotations, visualize_image_set

if len(sys.argv) != 2:
    print(__doc__)
    sys.exit(1)

data_path = os.path.abspath(sys.argv[1])
image_dir = os.path.join(data_path, "train/images")
annotations_path = os.path.join(data_path, "annotations/train.json")

annotations = get_annotations(annotations_path)
desired_categories = {'tennis-court','soccer-ball-field','ground-track-field','baseball-diamond'}
desired_ids = construct_desired_ids(desired_categories, annotations['categories'])
# construct dictionaries containing info about images
(images_dict, file_name_dict) = construct_dicts(annotations)
# create category index in the correct format for retraining and detection
category_index = construct_category_index(annotations, desired_categories)
label_id_offsets = calculate_label_id_offsets(category_index)

# set windowing information (size of window and stride); these values taken from DOTA paper
win_height = 1024
win_width  = 1024
win_stride_vert  = 512
win_stride_horiz = 512
win_set = (win_height, win_width, win_stride_vert, win_stride_horiz) # windowing information

import matplotlib
matplotlib.use("TkAgg")

preprocessor = Preprocessor(images_dict, file_name_dict, image_dir, annotations, category_index, win_set=win_set, min_coverage=.3)
for (window_np, box_set, class_set) in preprocessor.iterate():
    visualize_image_set([window_np],
            [box_set],
            [class_set],
            category_index,
            title=f"Window Batch", min_threshold=-1, interactive=False)
print(preprocessor.images_dict)
