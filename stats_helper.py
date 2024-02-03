import glob
import os
import numpy as np

from sklearn.preprocessing import StandardScaler

def compute_mean_and_std(dir_name: str) -> (np.array, np.array):
  '''
  Compute the mean and the standard deviation of the dataset.

  Note: convert the image in grayscale and then in [0,1] before computing mean
  and standard deviation

  Hints: use StandardScalar (check import statement)

  Args:
  -   dir_name: the path of the root dir
  Returns:
  -   mean: mean value of the dataset (np.array containing a scalar value)
  -   std: standard deviation of th dataset (np.array containing a scalar value)
  '''
  mean = None
  std = None

  ############################################################################
  # Student code begin
  ############################################################################
  
  im_paths = []
  for subdir, dir, files in os.walk(dir_name, topdown=True):
      for file in files:
          im_paths.append(os.path.join(subdir, file))
  im_paths = im_paths[2:]
  
  images = []
  for i in range(len(im_paths)):
    image = (np.asarray(Image.open(im_paths[i]).convert(mode="L"))).flatten()/255.0
    images.append(image)
  images = np.concatenate(images)

  scaler = StandardScaler()
  scaler.partial_fit(images.reshape(-1, 1))
  mean = scaler.mean_
  std = np.sqrt(scaler.var_)


  ############################################################################
  # Student code end
  ############################################################################
  return mean, std
