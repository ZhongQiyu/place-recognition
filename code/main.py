# import the modules on the drive after insertion
import os
from photo_manager import PhotoManager
from image_stack import ImageStack
from preprocessor import Preprocessor
from hog_base import HogDescriptor
import UnionBuildingRecognizer
from grid_search import GridSearcher
from validator import Validator


def main():
    """
    Driver function of the logistics module.
    # """
    photo_manager = PhotoManager(mode="local")
    repo = photo_manager.get_constants().REPO
    batch = os.path.join(repo, "unlabelled", "batch-53", "renamed")
    renamed = photo_manager.rename_all(batch)

    # initialize a photo manager for fitting the operations in the drive
    # photo_manager = PhotoManager()
    # update the photo manager's statistics
    # repo = photo_manager.get_constants().REPO
    # dset = photo_manager.get_constants().DSET
    # data_path = os.path.join(repo, dset)
    # photo_manager.set_stats(data_path)

    # enhance the data from the repository
    # if the instance is a video, then just improve the instances
    # with new videos
    # insts = os.listdir(batch)
    # for inst in insts:
    #     if const.MOV in inst:
    #         extracted = photo_manager.imagify(os.path.join(batch, inst))
    #         os.remove(os.path.join(batch, inst))

    # comment out for now, keep labelling until a point comes for automation
    # root = photo_manager.get_constants().ROOT
    # repo = os.path.join(root, "Concordiensis")
    # merged = "merged"
    # dest = os.path.join(root, merged)
    # src = [os.path.join(root, path) for path in os.listdir(root) if path.isdigit()]
    # merged_count = photo_manager.merge(src, dest)
    # print(merged_count)
    # repo = photo_manager.get_constants().REPO
    # unlabelled = os.path.join(repo, "unlabelled")
    # repeated = photo_manager.find_repeated dest)
    # missing = photo_manager.find_missing(dest, batch)


if __name__ == "__main__":
    main()
