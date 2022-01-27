import os
from error import OptionError


class Constants:
    """
    A Python class that stores all the constants to apply in the
    software.
    """
    def __init__(self, mode="web"):
        """
        Initialize the constants with a mode as the main parameter.
        The mode determines the path of the root as well as the path
        towards the data's repository.
        """
        self.mode = mode
        self.ROOT = self.assign_params(os.path.join("/content", "drive"),
                                       os.path.join("/Users", "allenzhong", "Downloads"))
        repo = os.path.join("Thesis", "dset")
        self.REPO = self.assign_params(os.path.join(self.ROOT, "MyDrive", repo),
                                       os.path.join(self.ROOT, repo))
        building_file = "campus-buildings.txt"
        building_info = self.assign_params(os.path.join(self.REPO, building_file),
                                           os.path.join(self.REPO, building_file))
        buildings = open(building_info).readlines()
        self.BUILDINGS = [building[building.find(" ") + 1:building.find("\n")] for building in buildings]
        self.DSET = "UnionBuildings"
        self.MISC = "Misc"
        self.CACHE = ".DS_Store"
        self.JPG = ".jpg"
        self.PNG = ".png"
        self.JPEG = ".jpeg"
        self.HEIC = ".heic"
        self.IMG = "IMG"
        self.MOV = ".MOV"
        self.BOUND = 10
        self.MAX_INDEX = 9999
        self.RGB_RANGE = 256
        self.IMG_LENGTH = 4032
        self.IMG_WIDTH = 3024
        self.CHANNELS = 3

    def get_mode(self):
        """
        Get the mode of the constants module, and return
        the mode in a textual manner.
        :return: the mode of the constants module in text.
        """
        return self.mode

    def set_mode(self, new_mode):
        """
        Set the new mode for the constants module.
        :param new_mode: the mode to set upon the constants module.
        """
        self.mode = new_mode

    def assign_params(self, web_params, local_params):
        """
        Assign parameters to the constants class so that they would become
        corresponding enough towards the usage upon other software modules.
        Currently there are only two modes, web and local; and if the user
        inputs a mode that does not belong to any of these categories, it
        would raise an error called OptionError.
        :param web_params: the parameter to pass when it is web mode.
        :param local_params: the parameter to pass when it is local mode.
        :return: the parameters based on the mode given by the user.
        """
        mode = self.get_mode()
        if mode == "web":
            return web_params
        elif mode == "local":
            return local_params
        else:
            raise OptionError
