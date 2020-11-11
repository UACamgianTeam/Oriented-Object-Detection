# 3rd Party
import tensorflow as tf
import numpy as np
# Python STL
from typing import Tuple, List, Set
from collections import defaultdict
import json
# Local
from .window import get_windows, get_window_ids, map_box_to_window
from ..utils.load_data import get_images


def preprocess_images(images_dict,
        file_name_dict,
        image_dir,
        annotations,
        category_index,
        win_set,
        verbose: bool = False):
    """
    Statetless: does not modify images_dict or file_name_dict

    :returns: A Generator of (images_dict, windows_np, gt_boxes, gt_classes, no_annotation_inds). The images_dict is the most-up-to-date version of the images_dict at the current iteration.
    """

    # Don't modify original version--Make a deep copy
    images_dict = _copy_json(images_dict)

    total_window_count = 0
    # start=1 because image_dict uses 1-based indexing
    for (image_id, image_np) in enumerate(get_images(image_dir, file_name_dict), start=1 ):
        (windows_dict, windows_np) = _slice_windows(image_np, win_set)
        # Ensures every window has a unique id
        windows_dict = { (k+total_window_count):v for (k,v) in windows_dict.items() }
        num_windows = len(windows_np)
        total_window_count += num_windows
        image_dict = images_dict[image_id]
        image_dict["windows"] = windows_dict
        image_dict["num_windows"] = len(image_dict["windows"])

        (image_dict_updates, windows_dict_updates) = \
                _map_annotations(image_id, image_dict, annotations, category_index)
        image_dict["boxes"].extend(image_dict_updates["boxes"])
        image_dict["classes"].extend(image_dict_updates["classes"])
        for (window_id, entry) in windows_dict_updates.items():
            windows_dict[window_id]["boxes"].extend(entry["boxes"])
            windows_dict[window_id]["classes"].extend(entry["classes"])
        (gt_boxes, gt_classes, no_annotation_ids) = _construct_gt(image_dict)
        yield (images_dict, windows_np, gt_boxes, gt_classes, no_annotation_ids)

def construct_category_index(train_annotations: dict, desired_categories: set) -> dict:
    """ Takes the category index from the training annotations and constructs it in the
    correct format to be used in the retraining process
    """
    # construct category index in correct format
    category_index  = {}
    for category in train_annotations['categories']:
        if category['name'] in desired_categories:
            category_index[category['id']] = {
                'id': category['id'],
                'name': category['name']
            }
    return category_index

def convert_to_tensors(train_images_np: List, gt_boxes: List, gt_classes: List, 
        label_id_offsets: dict, num_classes: int) -> Tuple:
    """ Converts class labels to one-hot; converts everything to tensors.

    The `label_id_offset` here shifts all classes by a certain number of indices;
    we do this here so that the model receives one-hot labels where non-background
    classes start counting at the zeroth index.  This is ordinarily just handled
    automatically in our training binaries, but we need to reproduce it here.
    """
    train_image_tensors = []
    gt_classes_one_hot_tensors = []
    gt_box_tensors = []
    for (train_image_np, gt_box_np, gt_class_np) in zip(train_images_np, gt_boxes, gt_classes):
        train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(
            train_image_np, dtype=tf.float32), axis=0))
        gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32))
        zero_indexed_groundtruth_classes = tf.convert_to_tensor(gt_class_np)
        gt_classes_one_hot_tensors.append(tf.one_hot(
            zero_indexed_groundtruth_classes, num_classes))

    return (train_image_tensors, gt_box_tensors, gt_classes_one_hot_tensors)


def map_category_ids_to_index(label_id_offsets: dict, category_ids_list: List) -> List:
    for index, category_ids in enumerate(category_ids_list):
        category_ids_list[index] = [label_id_offsets['map_to_index'][category_id] for category_id in category_ids_list[index]]
    return category_ids_list

def map_indices_to_category_ids(label_id_offsets: dict, indices: List) -> List:
        return [label_id_offsets['map_to_category'][index] for index in indices]

def calculate_label_id_offsets(category_index: dict) -> dict:
    """ for each category, calculate the offset we should give it to create an array
    of categories starting at index 0
    
    The retraining process wants categories in 0-index format starting at 0.
    """
    print(category_index)
    label_id_offsets = {
        'map_to_index': {},
        'map_to_category': {}
    }
    index = 0
    for category_id in category_index:
        label_id_offsets['map_to_index'][category_id] = index    # map from category_id -> index
        label_id_offsets['map_to_category'][index] = category_id # map from index -> category_id
        index += 1
    return label_id_offsets

def get_unsliced_images_np(image_dir: str, file_name_dict: dict) -> List:
    """ Returns original images (non-windowed) as an array of numpy images """
    unsliced_images_np = []
    for index, image_np in enumerate(get_images(image_dir, file_name_dict)):
        unsliced_images_np.append(image_np)
    return unsliced_images_np


# ************** Internal helper functions *****************


def _map_annotations(image_id : int,
                            image_dict : dict,
                            annotations : dict,
                            category_index : dict,
                            min_coverage : float = .7,
                            verbose : bool = False):
    """
    Stateless: does not modify image_dict

    :param image_id: The id for that image
    :param image_dict: The entry for that single image

    :returns: (image_dict_updates, window_dict_updates)
    """
    annotations_dict = annotations["annotations"]
    annotations_dict = filter(lambda a: a["image_id"] == image_id, annotations_dict)
    annotations_dict = filter(lambda a: a["category_id"] in category_index, annotations_dict)
    image_updates = {"boxes": [], "classes": []}
    window_updates = defaultdict(lambda: {"boxes": [], "classes": []})

    for annotation in annotations_dict:
        hbb = annotation['bbox'] # COCO stores annotations as [x,y,width,height]
        # calculate HBB in pure-coordinate format
        xmin = hbb[0]
        ymin = hbb[1]
        xmax = xmin + hbb[2]
        ymax = ymin + hbb[3]
        box = [ymin, xmin, ymax, xmax]
        image_updates["boxes"].append(box)
        image_updates["classes"].append(annotation["category_id"]) 
        windows_ids = get_window_ids(box, image_dict)
        for window_id in windows_ids:
            new_box = map_box_to_window(box, image_dict['dimensions'], image_dict['windows'][window_id])
            window_updates[window_id]["boxes"].append(new_box)
            window_updates[window_id]["classes"].append(annotation["category_id"])
    return (image_updates, window_updates)

def _construct_gt(image_dict) -> Tuple[List[np.ndarray],List[np.ndarray],Set[int]]:
    num_windows = len(image_dict["windows"])
    gt_boxes    = [None for _ in range(num_windows)]
    gt_classes  = [None for _ in range(num_windows)]
    no_annotation_ids = set()
    for (i, (win_id, window)) in enumerate(image_dict["windows"].items()):
        if window["boxes"]:
            gt_boxes[i] = np.array(window["boxes"],   dtype=np.float32)
            gt_classes[i] = np.array(window["classes"], dtype=np.int32)
            assert len(gt_boxes[i]) == len(gt_classes[i])
        else:
            no_annotation_ids.add(i)
    return (gt_boxes, gt_classes, no_annotation_ids)

def _slice_windows(image_np : np.ndarray, win_set : Tuple[int,int,int,int] = None) -> Tuple[List[np.ndarray], dict]:
    windows_np = []
    windows = dict()
    for win_index, (window, xmin, ymin, xmax, ymax) in enumerate(get_windows(image_np, *win_set, False)):
        window_dict = {}
        # window_dict['window'] = window
        window_dict['xmin'] = xmin
        window_dict['ymin'] = ymin
        window_dict['xmax'] = xmax
        window_dict['ymax'] = ymax
        window_dict['dimension_box'] = (ymin, xmin, ymax, xmax)
        window_dict['dimensions'] = (xmax - xmin, ymax - ymin)
        window_dict['boxes'] = []
        window_dict['classes'] = []
        windows[len(windows_np)] = window_dict # 0,1,2,3...# of windows - 1
        windows_np.append(window)
    return (windows, windows_np)

def _copy_json(root_obj):
    def hook(obj):
        def coerce_num(x):
            try: return int(x)
            except ValueError as e: pass
            try:                      return float(x)
            except ValueError as e:   pass
            return x
        return { coerce_num(k):v for (k,v) in obj.items() }
    return json.loads( json.dumps(root_obj), object_hook=hook)


__all__ = ["preprocess_images",
        "construct_category_index",
        "convert_to_tensors",
        "map_category_ids_to_index",
        "map_indices_to_category_ids",
        "calculate_label_id_offsets",
        "get_unsliced_images_np"]
