import json
import numpy as np
from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
from .cocoeval import COCOeval
from ..postprocess import percentage_to_coco

from typing import Tuple, List

# evaluate using pycocotools
annType = 'bbox'

def evaluate(data_path: str, test_images_dict: dict, desired_ids: set, 
        id_to_category: dict, min_threshold: float) -> None:
    print('Evaluations per window: ')
    evaluate_set(data_path, test_images_dict, desired_ids, id_to_category, min_threshold, 'window')
    print('\n')
    print('Evaluations per image: ')
    evaluate_set(data_path, test_images_dict, desired_ids, id_to_category, min_threshold, 'image')

def evaluate_set(data_path: str, test_images_dict: dict, desired_ids: set, 
        id_to_category: dict, min_threshold: float, set_name: str) -> None:
    annotation_path = data_path + '/annotations/'

    if set_name == 'image':
        labels_path  = annotation_path + 'validation.json'               # ground-truth
        results_path = annotation_path + 'evaluation/image_results.json' # predicted
        write_image_results(results_path, test_images_dict, min_threshold)
    elif set_name == 'window':
        labels_path  = annotation_path + 'validation_window.json'         # ground-truth
        results_path = annotation_path + 'evaluation/window_results.json' # predicted
        write_window_results(results_path, test_images_dict, min_threshold)
    cocoGt = COCO(labels_path)
    cocoDt = cocoGt.loadRes(results_path)

    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.catIds = list(desired_ids) # set category ids we want to evaluate on
    # cocoEval.params.useCats = [1]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize(id_to_category)

def write_image_results(results_path: str, test_images_dict: dict, min_threshold: float) -> None:
    # write results to a coco-format results file
    results = []
    for image_id, test_image_dict in test_images_dict.items():
        for index, predicted_box in enumerate(test_image_dict['predicted_boxes']):
        # for index, bbox in enumerate(test_image_dict['boxes']): # ground truth (should get 1.0 scores)
            # (xmin, ymin, xmax, ymax) = bbox
            # coco_bbox = (xmin, ymin, xmax - xmin, ymax - ymin) # coco bbox is (xmin, ymin, width, height)
            predicted_score = test_image_dict['predicted_scores'][index]
            predicted_class = test_image_dict['predicted_classes'][index]
            # convert from numpy arrays if necessary
            (predicted_box, predicted_class, predicted_score) = convert_from_np(
                predicted_box, predicted_class, predicted_score)
            # set detection resul if predicted score is high enough
            if test_image_dict['predicted_scores'][index] > min_threshold:
                detection = {
                    'image_id': image_id,
                    'category_id': predicted_class,
                    'bbox': percentage_to_coco(predicted_box, test_image_dict['dimensions']),
                    'score': predicted_score# 100.0
                }
                results.append(detection)

    with open(results_path, 'w') as outfile:
        json.dump(results, outfile)

def write_window_results(results_path: str, test_images_dict: dict, min_threshold: float) -> None:
    # # write results to a coco-format results file
    results = []
    for image_id, test_image_dict in test_images_dict.items():
        for window_id, test_window_dict in test_image_dict['windows'].items():
            for index, predicted_box in enumerate(test_window_dict['predicted_boxes']):
                predicted_score = test_window_dict['predicted_scores'][index]
                predicted_class = test_window_dict['predicted_classes'][index]
                # convert from numpy arrays if necessary 
                (predicted_box, predicted_class, predicted_score) = convert_from_np(
                    predicted_box, predicted_class, predicted_score)
                # set detection resul if predicted score is high enough
                if test_window_dict['predicted_scores'][index] > min_threshold:
                    detection = {
                        'image_id': window_id+1,
                        'category_id': predicted_class,
                        'bbox': percentage_to_coco(predicted_box, test_window_dict['dimensions']),
                        'score': predicted_score
                    }
                    results.append(detection)

    with open(results_path, 'w') as outfile:
        json.dump(results, outfile)

def write_window_validation_file(data_path: str, test_annotations: dict, test_images_dict: dict) -> None:
    """ Writes the validation json file for the windows in the test set """
    window_validation_file_path = data_path + '/annotations/validation_window.json'

    window_validation = {}
    window_validation['info'] = test_annotations['info']
    # construct a list of windows
    window_validation['images'] = []
    for image_id, test_image_dict in test_images_dict.items():
        for window_id, test_window_dict in test_image_dict['windows'].items():
            (width, height) = test_window_dict['dimensions']
            image = {
                'id': window_id+1,
                'file_name': 'window.png',
                'width': width,
                'height': height
            }
            window_validation['images'].append(image)
    window_validation['categories']  = [category for category in test_annotations['categories']]
    # map each ground truth annotation in the image set to their corresponding window(s)
    window_validation['annotations'] = []
    id = 1
    for image_id, test_image_dict in test_images_dict.items():
        for window_id, test_window_dict in test_image_dict['windows'].items():
            win_dimensions = test_window_dict['dimensions']
            (width, height) = win_dimensions
            for (box, category_id) in zip(test_window_dict['boxes'], test_window_dict['classes']):
                annotation = {
                    'area': width*height,
                    'category_id': category_id,
                    'bbox': percentage_to_coco(box, win_dimensions),
                    'image_id': window_id+1,
                    'iscrowd': 0,
                    'id': id
                }
                window_validation['annotations'].append(annotation)
                id += 1

    with open(window_validation_file_path, 'w') as outfile:
        json.dump(window_validation, outfile)

def convert_from_np(box, category, score) -> Tuple[List, int, float]:
    """ if any of the passed objects are numpy, convert to correct type """
    if isinstance(box, np.ndarray):
        box = box.tolist()
    if isinstance(category, np.uint32):
        category = category.item()
    if isinstance(score, np.float32):
        score = score.item()
    
    return (box, category, score)
    
