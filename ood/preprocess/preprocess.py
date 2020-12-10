# 3rd Party
import tensorflow as tf
import numpy as np
# Python STL
from typing import Tuple, List, Set
from collections import defaultdict
import json
# Local
from .window import get_windows, get_window_ids, map_box_to_window
from ..utils import copy_json
from ..utils.load_data import get_images

_DEFAULT_MIN_COVERAGE = .7

class Preprocessor(object):

    def __init__(self,
            images_dict,
            file_name_dict,
            image_dir,
            annotations,
            category_index,
            win_set=None,
            min_coverage=_DEFAULT_MIN_COVERAGE):

        self._images_dict    = _copy_json(images_dict)
        self._image_dir      = image_dir
        self._file_name_dict = file_name_dict
        self._annotations    = annotations
        self._category_index = category_index
        self._win_set        = win_set
        self._min_coverage   = min_coverage

        self._cur_window_id = None
        self._cur_image_id = None

    @property
    def images_dict(self):
        return self._images_dict

    @property
    def cur_image_id(self):
        return self._cur_image_id

    @property
    def cur_window_id(self):
        """
        Dictionary of the window that was just yielded
        """
        return self._cur_window_id
    
    def iterate(self, keep_empty_windows : bool  = False):
        """ Yields a "sliced window" for each window in each image. The images it will
        iterate over are found in image_dir.

        Each of the images are sliced into overlapping subsections called windows. These
        windows and their corresponding categories and ground truth boxes are yielded in
        a generator so that they are not all stored in memory at once.
        """
        images_dict    = self._images_dict
        image_dir      = self._image_dir
        file_name_dict = self._file_name_dict 
        annotations    = self._annotations    
        category_index = self._category_index 
        min_coverage   = self._min_coverage
        win_set        = self._win_set

        total_window_count = 0
        # start=1 because image_dict uses 1-based indexing
        for (image_id, image_np) in enumerate(get_images(image_dir, file_name_dict), start=1 ):
            (windows_dict, windows_np) = _slice_windows(image_np, win_set)
            # Ensures every window has a unique id

            for k in windows_dict:
                windows_dict[k]["window_array_index"] = k
            windows_dict = { (k+total_window_count):v for (k,v) in windows_dict.items() }


            num_windows = len(windows_np)
            total_window_count += num_windows
            image_dict = images_dict[image_id]
            image_dict["windows"] = windows_dict
            image_dict["num_windows"] = len(image_dict["windows"])
    
            (image_dict_updates, windows_dict_updates) = \
                    _map_annotations(image_id, image_dict, annotations, category_index, min_coverage=min_coverage)
            image_dict["boxes"].extend(image_dict_updates["boxes"])
            image_dict["classes"].extend(image_dict_updates["classes"])
            for (window_id, entry) in windows_dict_updates.items():
                windows_dict[window_id]["boxes"].extend(entry["boxes"])
                windows_dict[window_id]["classes"].extend(entry["classes"])

            sorted_ids = sorted(windows_dict.keys())
            for window_id in sorted_ids:
                window_dict = windows_dict[window_id]
                (gt_boxes,gt_classes, has_annotations) = _construct_gt_window(window_dict)
                if keep_empty_windows or has_annotations: 
                    arr_index = window_dict["window_array_index"]
                    self._cur_window_id = window_id
                    self._cur_image_id = image_id
                    window_np = windows_np[arr_index]
                    yield (window_np, gt_boxes, gt_classes)

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
    """ Maps a list of category ids to a 0-indexed array based on the
    predefined mapping (label_id_offsets)

    We must do this because training requires 0-indexed category ids.
    """
    if len(category_ids_list) == 1: # should return un-nested list
        return [label_id_offsets['map_to_index'][category_id] for category_id in category_ids]
    else:
        new_list = []
        for index, category_ids in enumerate(category_ids_list):
            new_list.append( [label_id_offsets['map_to_index'][category_id] for category_id in category_ids_list[index]] )
        return new_list

def map_indices_to_category_ids(label_id_offsets: dict, indices: List) -> List:
    """ Maps a list of 0-indexed indices back to their categories """
    return [label_id_offsets['map_to_category'][index] for index in indices]

def calculate_label_id_offsets(category_index: dict) -> dict:
    """ for each category, calculate the offset we should give it to create an array
    of categories starting at index 0
    
    The retraining process wants categories in 0-index format starting at 0.
    """
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
                            min_coverage : float = _DEFAULT_MIN_COVERAGE,
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
        windows_ids = get_window_ids(box, image_dict, min_coverage=min_coverage)
        for window_id in windows_ids:
            new_box = map_box_to_window(box, image_dict['dimensions'], image_dict['windows'][window_id])
            window_updates[window_id]["boxes"].append(new_box)
            window_updates[window_id]["classes"].append(annotation["category_id"])
    return (image_updates, window_updates)

def _construct_gt_window(window):
    if not window["boxes"]:
        return (None, None, False)
    gt_boxes   = np.array(window["boxes"], dtype=np.float32)
    gt_classes = np.array(window["classes"], dtype=np.int32)
    assert len(gt_boxes) == len(gt_classes)
    return (gt_boxes, gt_classes, True)

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


__all__ = ["Preprocessor",
        "construct_category_index",
        "convert_to_tensors",
        "map_category_ids_to_index",
        "map_indices_to_category_ids",
        "calculate_label_id_offsets",
        "get_unsliced_images_np"]
