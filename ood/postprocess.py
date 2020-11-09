import numpy as np
from nms import nms

from typing import List, Tuple

def restore_image_detections(test_images_dict: dict, min_threshold: float) -> (dict, List, List, List):
    """ Combine window results to restore detection results on original image 
    
    Also return lists of boxes and detections for visualization purposes
    """
    predicted_boxes   = [None] * len(test_images_dict)
    predicted_classes = [None] * len(test_images_dict)
    predicted_scores  = [None] * len(test_images_dict)
    for image_id, test_image_dict in test_images_dict.items():
        img_dimensions = test_image_dict['dimensions']
        predicted_boxes[image_id-1]   = []
        predicted_classes[image_id-1] = []
        predicted_scores[image_id-1]  = []
        for window_id, window_dict in test_image_dict['windows'].items():
            window_dimensions = window_dict['dimension_box']
            annotated_boxes   = window_dict['predicted_boxes']
            classes           = window_dict['predicted_classes']
            scores            = window_dict['predicted_scores']
            for index in range(len(annotated_boxes)):
                score = scores[index]
                if score > min_threshold:
                    image_box = map_box_to_image(tuple(annotated_boxes[index]), window_dimensions, img_dimensions)
                    class_category = classes[index]

                    test_images_dict[image_id]['predicted_boxes'].append(image_box)
                    test_images_dict[image_id]['predicted_classes'].append(class_category)
                    test_images_dict[image_id]['predicted_scores'].append(score)
                    predicted_boxes[image_id-1].append(image_box)
                    predicted_classes[image_id-1].append(class_category)
                    predicted_scores[image_id-1].append(score)
    
    return (test_images_dict, predicted_boxes, predicted_classes, predicted_scores)

def map_box_to_image(box: Tuple, win_dimensions: Tuple, img_dimensions: Tuple) -> List:
    """ Maps a box in a window to its original image

    Goes from [ymin%, xmin%, ymax%, xmax%] in window to [ymin%, xmin%, ymax%, xmax%] in image
    """
    (ymin_percentage, xmin_percentage, ymax_percentage, xmax_percentage) = box
    (win_ymin, win_xmin, win_ymax, win_xmax) = win_dimensions
    (img_width, img_height) = img_dimensions
    # convert percentage to absoluate position
    x_length = win_xmax - win_xmin
    y_length = win_ymax - win_ymin
    xmin = xmin_percentage * x_length
    xmax = xmax_percentage * x_length
    ymin = ymin_percentage * y_length
    ymax = ymax_percentage * y_length
    # convert from window coords -> image coords
    xmin += win_xmin
    xmax += win_xmin
    ymin += win_ymin
    ymax += win_ymin
    # normalize to get relative location of annotations
    xmin = xmin / img_width
    xmax = xmax / img_width
    ymin = ymin / img_height
    ymax = ymax / img_height
    return [ymin, xmin, ymax, xmax]

def run_nms(test_images_dict: dict) -> dict:
    """ Runs non-maximum suppression on all of the predicted boxes/scores for each of the images and
    all of the images' windows

    Stores the results in test_images dict

    This should prune any boxes that overlap considerably and it should favor the higher-scoring boxes
    """
    # run non-maximum suppression per image
    for image_id, image_info in test_images_dict.items():
        # convert boxes to coordinates that nms wants
        predicted_boxes = [percentage_to_coco(box, image_info['dimensions']) for box in image_info['predicted_boxes']]

        image_indices = nms.boxes(predicted_boxes, image_info['predicted_scores'])
        test_images_dict[image_id]['predicted_boxes']  = \
            [box for index, box in enumerate(image_info['predicted_boxes'])  if index in image_indices]
        test_images_dict[image_id]['predicted_classes']  = \
            [category for index, category in enumerate(image_info['predicted_classes'])  if index in image_indices]
        test_images_dict[image_id]['predicted_scores'] = \
            [score for index, score in enumerate(image_info['predicted_scores']) if index in image_indices]
        # run non-maxmium suppression per-window
        for window_id, window_info in image_info['windows'].items():
            # convert boxes to coordinates that nms wants
            predicted_boxes = [percentage_to_coco(box, window_info['dimensions']) for box in window_info['predicted_boxes']]

            window_indices = nms.boxes(predicted_boxes, window_info['predicted_scores'])
            test_images_dict[image_id]['windows'][window_id]['predicted_boxes'] = \
                [box for index, box in enumerate(window_info['predicted_boxes'])  if index in window_indices]
            test_images_dict[image_id]['windows'][window_id]['predicted_classes'] = \
                [category for index, category in enumerate(window_info['predicted_classes'])  if index in window_indices]
            test_images_dict[image_id]['windows'][window_id]['predicted_scores'] = \
                [score for index, score in enumerate(window_info['predicted_scores'])  if index in window_indices]

    return test_images_dict

def non_max_suppression(boxes: List, scores: List) -> List:
    """ Takes a list of boxes in the format (xmin,ymin,xmax,ymax) and a list of scores
    associated with those boxes and...
        (1) Changes format of boxes -> (xmin,ymin,width,height)
        (2) Runs non-maximum suppression on the boxes/scores
        (3) Returns a list of indices that should be kept
    """
    # boxes stored as (xmin,ymin,xmax,ymax) so need to map...
    for index, box in enumerate(boxes):
        (ymin, xmin, ymax, xmax) = box
        boxes[index] = (xmin,ymin,xmax-xmin,ymax-ymin)
    # takes boxes in form (x,y,w,h)
    return nms.boxes(boxes, scores)

def percentage_to_coco(box: Tuple, img_dimensions: Tuple) -> Tuple:
    """ Maps a box in the relative coords format (ymin%, xmin%, ymax%, xmax%)
    -> (xmin, ymin, width, height) absolute coords format
    """
    (ymin_percentage, xmin_percentage, ymax_percentage, xmax_percentage) = box
    (width, height) = img_dimensions
    # convert percentage to absoluate position
    xmin = xmin_percentage * width
    xmax = xmax_percentage * width
    ymin = ymin_percentage * height
    ymax = ymax_percentage * height
    # COCO format is (xmin, ymin, width, height)
    return [xmin, ymin, xmax - xmin, ymax - ymin]