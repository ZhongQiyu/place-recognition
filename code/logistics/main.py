import os
from photo_manager import PhotoManager


def main():
    photo_manager = PhotoManager()
    src = photo_manager.LOCAL_REPO

    # comment out for now, keep labelling until a
    # point comes to be necessary for automation
    # root = LOCAL_ROOT
    # repo = os.path.join(root, "Concordiensis")
    # renamed = photo_manager.rename_photos(repo)

    merged = "merged"
    dest = os.path.join(src, merged)
    merged_count = photo_manager.merge(src, dest)
    batch = "batch-()"  # rename
    repeated = photo_manager.find_repeated(dest)
    missing = photo_manager.find_missing(dest, batch)


if __name__ == "__main__":
    main()
