import os

WEB_ROOT = os.path.join("/content", "drive")
LOCAL_ROOT = os.path.join("/Users", "allenzhong", "Downloads")
REPO = os.path.join("Thesis", "dset")
WEB_REPO = os.path.join(WEB_ROOT, "MyDrive", REPO)
LOCAL_REPO = os.path.join(LOCAL_ROOT, REPO)
DSET = "UnionBuildings"
BUILDINGS = [building[building.find(" ") + 1:building.find("\n")]
             for building in open(os.path.join(LOCAL_REPO[:LOCAL_REPO.rfind("/")], "campus-buildings.txt")).readlines()]
MISC = "Misc"
CACHE = ".DS_Store"
# may want to generalize the file formats
JPEG = ".jpeg"
HEIC = ".heic"
# what are these two below for
POS_CURRENT = ...
NEG_CURRENT = ...
MAX_INDEX = 9999
RGB_RANGE = 256
IMG_LENGTH = 4032
IMG_WIDTH = 3024
CHANNELS = 3
