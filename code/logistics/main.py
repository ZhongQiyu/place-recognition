import os
from photo_manager import PhotoManager


def main():
    """
    Driver function of the logistics module.
    """
    photo_manager = PhotoManager(mode="local")
    root = photo_manager.get_constants().ROOT
    repo = photo_manager.get_constants().REPO
    batch = os.path.join(repo, "unlabelled", "batch-45", "renamed")
    renamed = photo_manager.rename_all(batch)

    # enhance the data from the repository
    # if the instance is a video, then just improve the instances
    # with new videos
    insts = os.listdir(batch)
    for inst in insts:
        if const.MOV in inst:
            extracted = photo_manager.imagify(os.path.join(batch, inst))
            os.remove(os.path.join(batch, inst))
    # comment out for now, keep labelling until a point comes to be necessary for automation
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
