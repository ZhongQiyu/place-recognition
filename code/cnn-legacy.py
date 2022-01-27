import cv2
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# colors


# define a function to load, transpose (if needed), and normalize the image data
def process_img(path, ext_channels=False):
  """
  Process an image from the dataset of UnionBuildings.
  :param path: the path on Google Drive that stores the image.
  :param ext_channels: indicate whether the channel data is returned or not.
  :return: the data of the given processed image. if to extract the channel data,
           the mass won't be normalized; otherwise, the data would be normalized.
  """
  # extract the image's index
  img_index = path[path.find(".jpeg") - IMG_IDX_LEN:path.find(".jpeg")]

  # load with RGB
  photo_data = cv2.imread(path)

  # load with grayscale
  # EXPENSIVE OP #1
  # photo_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

  # linear regression: flatten the data, then convolute the whole image?
  # CNN: adjust the filter size when it comes to different dimensions

  # press the VisibleDeprecationWarning
  np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

  # normalize the photos as grayscale if do not extract the features
  # EXPENSIVE OP #2
  # if not ext_channels:
  #   stdscaler = StandardScaler()
  #   photo_data = stdscaler.fit_transform(photo_data)
  # else:
  #   photo_data = cv2.cvtColor(photo_data, cv2.COLOR_GRAY2RGB)

  return photo_data

# add the data points from an external data sheet
# *transform to be something more generic, if needed

# parse the image datasheet
# sheet = photo_manager.parse_dset(ROOT + THESIS_DIR + DSET_FILE, option="3")

# modify to fit your system - see code below
# data_path = fr'{os.getenv("HOME")}/downloads/animalface/Image'
# os.listdir(data_path)

# STEP 1.1: design a single function that returns the color-channel data

# based on Wikipedia, R and G channels are for recognizing the details,
# whereas the B channel is for adding a feature over the environment


def ext_color(photo_data, color):
  """
  Extract a color channel's data in an image,
  given the path of the image and the color
  of the channel to extract things from.
  :param photo_data: the data of the photo.
  :param color: the channel to extract color from the photo's data.
  :return: the data of the selected color channel.
  """
  sliced_data = None

  if color.upper() == "R":
    sliced_data = photo_data[:, :, 0]

  elif color.upper() == "G":
    sliced_data = photo_data[:, :, 1]

  elif color.upper() == "B":
    sliced_data = photo_data[:, :, -1]

  # consider adding the opacity channel

  else:  # color.upper() is "B"; add exception modules later
    print("I am sorry, but there is some error with your input of color.")

  return sliced_data


def parse_color(color_data):
  """
  Use the idea of color space, parse the color data of a single
  image by averaging out the data in that color channel. Return
  this average value to represent the image, for segmentation.
  :param color_data: the data of a color channel, either R, G, or B.
  :return: the color of that channel by the retrieving the average.
  """
  return float("{:.2f}".format(np.mean(color_data)))

# STEP 1.2: transform the original photo data


def conv_color(photo_data):
  """
  Perform convolutional operations on the colored images to
  extract the channel data regarding the dataset to transform
  into. The window should be applied upon a processed, colored
  image, using () method.
  :param photo_data: the data of the colored image to perform convolution on.
  :return: a tuple that contains the averaged R, G, and B values from the image.
  """
  # extract the channeled data
  r_data = ext_color(photo_data, "R")
  g_data = ext_color(photo_data, "G")
  b_data = ext_color(photo_data, "B")

  # convolute the channeled data
  r_value = parse_color(r_data)
  g_value = parse_color(g_data)
  b_value = parse_color(b_data)

  return r_value, g_value, b_value

# Step 1.3.a: unit testing with one of the images in the Google Drive

# IMG_PATH = "IMG_xxxx" + JPEG  # generalize w/ string methods if needed in CNN
# IMG_IDX_LEN = IMG_PATH.count("x")  # check if still needed in CNN

# construct the path of the image
img_path = ROOT + THESIS_DIR + DSET_DIR + IMG_PATH.replace("xxxx", "8196")

# process to retrieve the data of the photo
photo_data = process_img(img_path,ext_channels=True)

# get the averaged channel data
r, g, b = conv_color(photo_data)

# cv2_imshow(cv2.resize(photo_data, (int(IMG_HEIGHT * SCALE_CONST), int(IMG_WIDTH * SCALE_CONST))))
print(f"The average R value is {r}, average G is {g}, and average B is {b}, " +
      f"for the image {img_path[img_path.rfind('/')+1:]}.")

# Step 1.3.b: unit testing with all the images in the Google Drive

# read the dataset from the directory
base_path = ROOT + THESIS_DIR + DSET_DIR
img_dir = [img_name for img_name in listdir(base_path) if img_name.find(".jpeg") != -1]
inst_count = len(img_dir)
# new_dset = "/img_rgb.csv"
new_data = []

# iterate through the directory
for img in img_dir:

  # load and read the data
  img_path = base_path + "/" + img
  photo_data = process_img(img_path)
  r, g, b = conv_color(photo_data)
  new_data.append([r, g, b])

  # with open(base_path + new_dset, "w", newline="") as rgb_data:
  #   rgb_writer = csv.writer(rgb_data, delimiter=",")
  #   rgb_writer.writerow([r, g, b])

# with open(base_path + new_dset) as rgb_data:
#   rgb_reader = csv.reader(rgb_data)
#   for rgb in rgb_reader:
#     print(rgb)

# update the data format
new_data = np.array(new_data)

# test with the first instance
print(new_data[0])

# *attempt of visualization
r, g, b = cv2.split(photo_data)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

pixel_colors = photo_data.reshape((IMG_HEIGHT*IMG_WIDTH, CHANNELS))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker="o")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()


# step 2.1: angle of the building
def ext_angle(path):
  """
  Extract the angle feature
  """

# step 3.1: pattern of the building (i.e., edges?)

# analyticsvidhya.com/blog/2021/06/feature-detection-description-and-matching-of-images-using-opencv/

# step 4.1: shape of the building (i.e., edges again?)

# parse the data of a single image into a new .csv,
# containing the features of color, pattern, illumination,
# angle, shape, etc.
# This should be a method in UnionBuildingRecognizer.


def update_data(data_file, attribute):
  """
  Update the data sheet where all the data of the attributes are stored.
  Return the updated data sheet.
  :param data_file: the file of the data to update
  """

# web crawling: API of Union's Instagram Account

# web crawling: API to Union's Flicker Account
