import tensorflow as tf
import numpy as np
from typing import Tuple, List
import json

from .window import get_windows, get_window_ids, map_box_to_window

from ..utils.load_data import get_images
#from utils.load_data import get_images


# ********** Ethan's heavy refactoring **********

def eth_preprocess_images(images_dict,
        file_name_dict,
        image_dir,
        annotations,
        category_index,
        win_set,
        verbose: bool = False):

    # So I can still use normal naming conventions internallly
    slice_windows = eth_slice_windows

    # Don't modify original versions
    images_dict = json.loads( json.dumps(images_dict) )
    file_name_dict = json.loads( json.dumps(file_name_dict) )

    for (image_index, image_np) in enumerate(image_dir, file_name_dict):
        (window_dict, windows_np) = slice_windows(image_np, win_set)
        pass

def ethan_filter_annotations():
    pass

def eth_slice_windows(image_np : np.ndarray, win_set : Tuple[int,int,int,int] = None) -> Tuple[List[np.ndarray], dict]:
    window_dict = dict()
    windows_np = []
    for index, image_np in enumerate(get_images(train_image_dir, file_name_dict)):
        # divide image into windows of size win_height * win_width 
        # and add to dictionary corresponding to that image
        windows = {}
        for win_index, (window, xmin, ymin, xmax, ymax) in enumerate(get_windows(image_np, *win_set, False)):
            if verbose: print('NEW WINDOW with dimensions ' + str(window.shape))
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
    return window_dict

# ********** external functions *****************


def preprocess_train_images(images_dict: dict, file_name_dict: dict, train_image_dir: str, 
                            train_annotations: dict, category_index: dict,
                            win_set: Tuple[int,int,int,int], verbose: bool) -> Tuple[List, List, List]:
    """ Preprocesses the training images into a format that the model can be retrained on
    1. constructs a list of all of the windows in the training set and map them to their corresponding image (images_dict)
    2. maps each annotation to its corresponding window(s)
    3. constructs the ground truth boxes for each of the training windows and gets dict of windows with no associated annotations
    4. removes windows and their corresponding "boxes" in which there are no annotations

    * train_images_np: a list of the windows that will be used to retrain the model
    * gt_boxes: The ground truth boxes for each of the windows in train_images_np
    """
    # construct a list of all of the windows in the training set and map them to their corresponding image (images_dict)
    (train_images_np, images_dict) = construct_images_np(images_dict, file_name_dict, train_image_dir, verbose, win_set)
    # map each annotation to its corresponding window(s) where annotation = (box, class) in different lists
    images_dict = map_annotations(images_dict, train_annotations, category_index, verbose)
    # construct the ground truth boxes & classes for each of the training windows and get dict of windows with no associated annotations
    (gt_boxes, gt_classes, no_annotation_ids) = construct_gt(images_dict, len(train_images_np), verbose)
    # remove windows and their corresponding "boxes" in which there are no annotations
    # remove in-place using list comprehensions to avoid copying large arrays
    train_images_np[:] = [train_image_np for index, train_image_np in enumerate(train_images_np) if not index in no_annotation_ids]
    gt_boxes[:] = [gt_box for index, gt_box in enumerate(gt_boxes) if not index in no_annotation_ids]
    gt_classes[:] = [gt_class for index, gt_class in enumerate(gt_classes) if not index in no_annotation_ids]

    return (train_images_np, gt_boxes, gt_classes)

def preprocess_validation_loss_images(test_images_dict: dict, file_name_dict: dict, test_image_dir: str,
        test_annotations: dict, category_index: dict,
        win_set: Tuple[int,int,int,int], verbose: bool) -> Tuple[List, List, List, dict]:
    """ This dataset is used  

    """
    # construct a list of all of the windows in the training set and map them to their corresponding image (images_dict)
    (test_images_np, test_images_dict) = construct_images_np(test_images_dict, file_name_dict, test_image_dir, verbose, win_set)

    # map each annotation to its corresponding window(s)
    test_images_dict = map_annotations(test_images_dict, test_annotations, category_index, verbose)

    # construct the ground truth boxes for each of the training windows and get dict of windows with no associated annotations
    (gt_boxes, gt_classes, no_annotation_ids) = construct_gt(test_images_dict, len(test_images_np), verbose)

    return (test_images_np, gt_boxes, gt_classes, test_images_dict, no_annotation_ids)


def preprocess_test_images(test_images_dict: dict, file_name_dict: dict, test_image_dir: str,
        test_annotations: dict, category_index: dict,
        win_set: Tuple[int,int,int,int], verbose: bool) -> Tuple[List, List, List, dict, List, List, List]:
    """ Preprocesses the validation set. We need two different datasets - one for computing the validation loss during the
    retraining step and one for evaluating the performance of the model.

    1. Validation loss dataset: Does not contain any images with no corresponding annotations
    2. Test dataset: Contains all windowed images in the test set regardless of whether they have annotations or not

    """
    # construct a list of all of the windows in the training set and map them to their corresponding image (images_dict)
    (test_images_np, test_images_dict) = construct_images_np(test_images_dict, file_name_dict, test_image_dir, verbose, win_set)

    # map each annotation to its corresponding window(s)
    test_images_dict = map_annotations(test_images_dict, test_annotations, category_index, verbose)

    # construct the ground truth boxes for each of the training windows and get dict of windows with no associated annotations
    (gt_boxes, gt_classes, no_annotation_ids) = construct_gt(test_images_dict, len(test_images_np), verbose)

    # remove windows and their corresponding "boxes" in which there are no annotations
    # remove in-place using list comprehensions to avoid copying large arrays
    valid_images_np = [valid_image_np for index, valid_image_np in enumerate(test_images_np) if not index in no_annotation_ids]
    valid_gt_boxes = [gt_box for index, gt_box in enumerate(gt_boxes) if not index in no_annotation_ids]
    valid_gt_classes = [gt_class for index, gt_class in enumerate(gt_classes) if not index in no_annotation_ids]

    # TODO - wrap these (image, box, class) tuples in a dataset wrapper like in retrain.py

    return (test_images_np, gt_boxes, gt_classes, test_images_dict, valid_images_np, valid_gt_boxes, valid_gt_classes)

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


# ************** Internal helper functions *****************

def construct_images_np(images_dict: dict, file_name_dict: dict, train_image_dir: str, 
        verbose: bool, win_set: Tuple[int,int,int,int]) -> Tuple[List, dict]:
    """ Takes each image in the given dictionary of images, breaks them up into windows, maps the window
    information to the dictionary for the image (images_dict), and adds each window to a list of numpy
    images (images_np)
    """
    images_np = [] # the sliding windows of the images
    # construct images_np - should be a list of all the windows of all the images
    for index, image_np in enumerate(get_images(train_image_dir, file_name_dict)):
        # divide image into windows of size win_height * win_width 
        # and add to dictionary corresponding to that image
        windows = {}
        for win_index, (window, xmin, ymin, xmax, ymax) in enumerate(get_windows(image_np, *win_set, False)):
            if verbose: print('NEW WINDOW with dimensions ' + str(window.shape))
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
            windows[len(images_np)] = window_dict # 0,1,2,3...# of windows - 1
            images_np.append(window)
    
        images_dict[index+1]['num windows'] = len(windows)
        images_dict[index+1]['windows'] = windows

    return (images_np, images_dict)

def get_unsliced_images_np(image_dir: str, file_name_dict: dict) -> List:
    """ Returns original images (non-windowed) as an array of numpy images """
    unsliced_images_np = []
    for index, image_np in enumerate(get_images(image_dir, file_name_dict)):
        unsliced_images_np.append(image_np)
    return unsliced_images_np

def construct_gt(images_dict: dict, num_windows: int, verbose: bool) -> Tuple[List, List, dict]:
    """ Constructs a list of the ground truth boxes to be used in the retraining step. This list
    corresponds to the training image set so that for each index in train_images_np, there is a
    corresponding index in gt_boxes that contains a list of all of the gt_boxes in that image
    (in our case, each of the windows).

    We also keep track of all of the windows in which there are no annotations associated with
    them so that we can remove these from the retraining set later.
    """
    # construct gt_boxes array of numpy lists of boxes for each window in each image
    gt_boxes = [None] * num_windows
    gt_classes = [None] * num_windows
    no_annotation_ids = {}
    for image_id in images_dict:
        # print('analyzing image ' + str(image_id))
        for win_id, window in images_dict[image_id]['windows'].items():
            # print(window['boxes'])
            if win_id < num_windows:
                gt_boxes[win_id] = np.array(window['boxes'], dtype=np.float32)
                gt_classes[win_id] = np.array(window['classes'], dtype=np.int32)
            # print(window['boxes'])
            if not window['boxes']:
                no_annotation_ids[win_id] = ''

    num_annotated_windows = num_windows - len(no_annotation_ids)
    percent_annotated_windows = round((num_annotated_windows / num_windows)*100)
    if verbose:
        print(str(percent_annotated_windows) + '% of windows have annotations associated with them.')
        print('This is ' + str(num_annotated_windows) + ' windows out of ' + str(num_windows))

    return (gt_boxes, gt_classes, no_annotation_ids)

def map_annotations(images_dict: dict, annotations: dict, category_index: dict, 
        verbose: bool) -> dict:
    """ Maps the image annotations to their corresponding images and windows. Note that the same 
    annotation can be mapped to multiple windows as the windows may overlap.

    We map an annotation if at least 70% of the annotation is preserved
    """
    num_successes = 0
    num_failures  = 0

    # Get annotated boxes for each image window
    annotations_dict = annotations['annotations']
    for annotation in annotations_dict:
        if annotation['category_id'] in category_index:
            # **** DOTA annotation info ****
            # Two forms of annotation exist in this annotations dictionary
            #   (1) Arbitrary BB: {(xi,yi) for i = 1,2,3,4}
            #   (2) HBB: [xmin, ymin, width, height]
            # We use (2) here because it more closely fits the original BB format
            hbb = annotation['bbox'] # COCO stores annotations as [x,y,wdith,height]
            # calculate HBB in new format
            xmin = hbb[0]
            ymin = hbb[1]
            xmax = xmin + hbb[2]
            ymax = ymin + hbb[3]
            box = [ymin, xmin, ymax, xmax]
            if verbose: print('annotation: ' + str(box))
            # map annotation to image
            image_id = annotation['image_id']
            images_dict[image_id]['boxes'].append(box)
            images_dict[image_id]['classes'].append(annotation['category_id'])
            # get corresponding window ids associated with the annotated box
            window_ids = get_window_ids(box, images_dict[image_id])
            if not window_ids:
                num_failures += 1
                if verbose: print('Couldn\'t find window for annotation ' + str(box) + ' in image ' + images_dict[image_id]['name'])
            else:
                if verbose: print('annotation ' + str(annotation['category_id']) + ' corresponds to ' + str(window_ids) + ' windows in image: ' + images_dict[image_id]['name'])
                num_successes += 1
                for window_id in window_ids:
                    if verbose: print('mapping box to window ' + str(window_id))
                    new_box = map_box_to_window(box, images_dict[image_id]['dimensions'], images_dict[image_id]['windows'][window_id])
                    if verbose: print('new box is ' + str(new_box))
                    images_dict[image_id]['windows'][window_id]['boxes'].append(new_box)
                    images_dict[image_id]['windows'][window_id]['classes'].append(annotation['category_id'])

    if verbose:
        print('Num successes: ' + str(num_successes))
        print('Num failures: ' + str(num_failures))
        print('Percent of annotations preserved: ' + str(round((num_successes / (num_successes + num_failures))*100,2)) + '%')

    return images_dict

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

def map_category_ids_to_index(label_id_offsets: dict, category_ids_list: List) -> List:
    for index, category_ids in enumerate(category_ids_list):
        category_ids_list[index] = [label_id_offsets['map_to_index'][category_id] for category_id in category_ids_list[index]]
    return category_ids_list

def map_indices_to_category_ids(label_id_offsets: dict, indices: List) -> List:
        return [label_id_offsets['map_to_category'][index] for index in indices]
