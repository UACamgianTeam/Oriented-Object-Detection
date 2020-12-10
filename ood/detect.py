import tensorflow as tf
import numpy as np

from .utils import copy_json
from .preprocess import map_indices_to_category_ids

from typing import List, Tuple

def run_inference(model, preprocessor, label_id_offsets):
    """ The given model performs inference on each window contained in the given
    Preprocessor class. The resulting bounding boxes, classes, and scores for
    each window are stored and returned in images_dict. Additionally, these
    boxes, classes, and scores are returned as lists.
    """
    p = preprocessor
    images_dict = p.images_dict

    predicted_boxes_list   = list()
    predicted_classes_list = list()
    predicted_scores_list  = list()

    prev_window_id = -1
    for (window_np, _, _) in p.iterate(keep_empty_windows=True):
        window_id = p.cur_window_id
        if window_id != prev_window_id + 1:
            raise Exception("run_inference expects Preprocessor.iterate to yield windows in \
              order of their id, with the ids starting at 0 and increasing by 1 with each iteration")
        prev_window_id = window_id
        image_id  = p.cur_image_id
        window_dict = p.images_dict[image_id]["windows"][window_id]
        input_tensor = tf.convert_to_tensor(np.expand_dims(window_np, axis=0), dtype=tf.float32)
        detections = detect(input_tensor, model)

        # set predicted detections
        predicted_boxes   = detections['detection_boxes'][0].numpy()
        predicted_scores = detections['detection_scores'][0].numpy()
        # we need to map the indices used for the retraining back to their original category ids
        predicted_classes = map_indices_to_category_ids(label_id_offsets, 
            detections['detection_classes'][0].numpy().astype(np.uint32).tolist())
        # set the results
        predicted_boxes_list.append(predicted_boxes)
        predicted_classes_list.append(predicted_classes)
        predicted_scores_list.append(predicted_scores)

        images_dict[image_id]['windows'][window_id]['predicted_boxes']   = predicted_boxes
        images_dict[image_id]['windows'][window_id]['predicted_classes'] = predicted_classes
        images_dict[image_id]['windows'][window_id]['predicted_scores']  = predicted_scores

    return (images_dict, predicted_boxes_list, predicted_classes_list, predicted_scores_list)

# Again, uncomment this decorator if you want to run inference eagerly
@tf.function(experimental_relax_shapes=True)
def detect(input_tensor: any, detection_model: any) -> dict:
  """Run detection on an input image.

  Args:
    input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
      Note that height and width can be anything since the image will be
      immediately resized according to the needs of the model within this
      function.

  Returns:
    A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
      and `detection_scores`).
  """
  preprocessed_image, shapes = detection_model.preprocess(input_tensor)
  prediction_dict = detection_model.predict(preprocessed_image, shapes)
  return detection_model.postprocess(prediction_dict, shapes)
