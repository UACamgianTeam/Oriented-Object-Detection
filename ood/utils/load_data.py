import os
import numpy as np
import tensorflow as tf
import json
from six import BytesIO
from PIL import Image

from typing import Generator

def copy_json(root_obj):
    def hook(obj):
        def coerce_num(x):
            try: return int(x)
            except ValueError as e: pass
            try:                      return float(x)
            except ValueError as e:   pass
            return x
        return { coerce_num(k):v for (k,v) in obj.items() }
    return json.loads( json.dumps(root_obj), object_hook=hook)

def get_images(image_dir: str, file_name_dict: dict) -> Generator[ np.ndarray, None, None ]:
  """ A generator for images given the image directory; yields images as numpy arrays
  in the order of the file names in the file name dictionary
  
  Yields an image as a numpy array
  """
  image_names = os.listdir(image_dir) # get a list of the names of the training images
  num_images = len(image_names) # get the total number of training images
  # convert images to numpy arrays
  for id, file_name in file_name_dict.items():
    # print(file_name)
    image_path = os.path.join(image_dir, file_name)
    # print(image_path + ' -> ' + str(id))
    image = np.array([])
    try:
      image = load_image_into_numpy_array(image_path)
    except ValueError as e: # if we can't load image, print error and don't use it.
      print('could not load image at ' + image_path)
      print(e)
    
    if image.size != 0: # if successfully loaded image
      yield load_image_into_numpy_array(image_path)

def get_annotations(annotation_path: str) -> dict:
  """ Loads the annotation file at the given path """
  try:
    with open(annotation_path) as json_file:
        annotations = json.load(json_file)
  except FileNotFoundError:
    raise Exception('Could not find file at ' + annotation_path)
  
  return annotations

def get_train_images(image_dir: str, file_name_dict: dict) -> Generator[ np.ndarray, None, None ]:
  """ A generator for images given the image directory 
  
  Yields an image as a numpy array
  """
  image_names = os.listdir(image_dir) # get a list of the names of the training images
  num_images = len(image_names) # get the total number of training images
  # convert images to numpy arrays
  for id, file_name in file_name_dict.items():
    # print(file_name)
    image_path = os.path.join(image_dir, file_name)
    # print(image_path + ' -> ' + str(id))
    yield load_image_into_numpy_array(image_path)

def get_test_images(image_dir: str) -> Generator[ np.ndarray, None, None ]:
  """ A generator for images given the image directory 
  
  Yields an image as a numpy array
  TODO - merge get_train_images and get_test_images into one function...
  """
  image_names = os.listdir(image_dir) # get a list of the names of the training images
  # print(image_names)
  num_images = len(image_names) # get the total number of training images
  # convert images to numpy arrays
  for index, image_name in enumerate(image_names):
    image_path = os.path.join(image_dir, image_name)
    yield load_image_into_numpy_array(image_path)

def load_image_into_numpy_array(path: str) -> Image:
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path.

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  image_np = np.array(image)

  if len(image_np.shape) == 2: # if grayscale, add RGB channels
    image_np = np.repeat(image_np[..., np.newaxis], 3, -1)
    # print('Added channels to grayscale image. New shape is: ' + str(image_np.shape))

  return image_np.reshape(
      (im_height, im_width, 3)).astype(np.uint8)
