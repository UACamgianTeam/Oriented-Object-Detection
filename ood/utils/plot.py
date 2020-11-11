import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from typing import List

from object_detection.utils import visualization_utils as viz_utils

min_threshold = 0.5 # paper uses 0.3 but I found 0.5 to work best here

def visualize_image_set(images_np: List, boxes_list: List, classes_list: List, 
    category_index: dict, title: str, min_threshold: float, scores_list = [], interactive=True) -> None:
  """ Displays the first 16 images and their corresponding annotations for the given 
  image set data 
  """
  should_use_dummy_scores = False
  if isinstance(scores_list, list) and not scores_list:
    should_use_dummy_scores = True

  if len(images_np) > 16:
    num_images_to_plot = 16
  else:
    num_images_to_plot = len(images_np)

  plt.figure(figsize=(30, 15))
  plt.suptitle(title)
  for idx in range(num_images_to_plot):
    boxes = boxes_list[idx]
    categories = classes_list[idx]
    if should_use_dummy_scores: # set scores to equal 100%
      scores = [1.0] * len(boxes)
    else:
      scores = scores_list[idx]
    print(scores)

    if not isinstance(boxes, np.ndarray):
      boxes = np.array(boxes)
    if not isinstance(scores, np.ndarray):
      scores = np.array(scores)
    
    plt.subplot(4, 4, idx+1)
    
    print('scores > ' + str(int(min_threshold*100)) + '%:')
    print([score for score in scores if score > min_threshold])

    plot_detections(
        images_np[idx],
        boxes,
        categories,
        scores, 
        category_index)
  if interactive: plt.ion()
  plt.show()
  # plt.savefig('tests/test_set.png')
  plt.pause(0.001)
  input('Press [enter] to continue.')

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
  """Wrapper function to visualize detections.

  Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
  """
  image_np_with_annotations = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_annotations,
      boxes,
      classes,
      scores,
      category_index,
      use_normalized_coordinates=True,
      min_score_thresh=min_threshold) # paper uses 0.3 i believe
  if image_name:
    plt.imsave(image_name, image_np_with_annotations)
  else:
    plt.imshow(image_np_with_annotations)
