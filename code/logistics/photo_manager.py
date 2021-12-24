import shutil
import os
import cv2
import matplotlib.pyplot as plt
from constants import *
from error import OptionError


# consider inheritance from an original Python library
class PhotoManager():
    """
    A photo manager that handles all the available photos as raw files,
    ensuring the pipeline for training the recognizer would work.
    """

    # take care of the warning on stats: it does matter when the module extends to software
    # especially during unit testing
    def __init__(self, mode="drive", stats={}):
        """
        Initialize a photo manager for the photo repository,
        so that the data would be ready for training. This
        initializer allows passing in a map of statistics
        over the photos.
        :param mode: the mode to initialize the photo manager.
        :param stats: the statistics to store the data in the photo manager.
        """
        # attributes
        self.mode = mode
        self.stats = stats
        # self.data = data
        if mode == "drive":
            self.root = os.path.join("content", "drive")
        elif mode == "local":
            self.root = os.path.join("Users", "allenzhong", "Downloads", "Thesis")
        else:
            raise OptionError

    def get_stats(self):
        """
        Return the statistics of the photos stored in the recognizer.
        """
        return self.stats

    def set_stats(self, new_stats):
        """
        Set the statistics of the photos, so that they will be appropriately
        updated.
        :param new_stats: the new map of statistics
        """
        self.stats = new_stats

    # *may need to generalize: assumes the data structure of self.stats is dictionary
    def update_stats(self, name, count):
        """
        Set the statistics of the photos stored in the recognizer,
        with any new form of the data.
        :param name: the name of the building, attached with the photos.
        :param count: the count of the photos.
        """
        self.stats.update({name: count})

    def show_stats(self, option="plain"):
        """
        Display the statistics of photos in a more formatted way,
        with the help of the attribute itself.
        :param option: plain for plain text, plot for matplotlib.pyplot.plot()
        """
        stats = self.get_stats()
        if option == "plain":
            # plain text
            if len(stats) != 0:
                for name, count in stats.items():
                    print(f"There are {count} instance(s) for {name}.")
                print(f"There are {sum(list(stats.values()))} instance(s) in total.")
            else:
                print("Currently, there is no data for the statistics.")
        elif option == "plot":
            # bar plot
            buildings = self.get_stats().keys()
            buildings_ab = ["".join([char for char in building if char.isupper()]) for building in buildings]
            counts = self.get_stats().values()
            fig = plt.figure(figsize=(8, 4))
            plt.bar(buildings_ab, counts, color='maroon', width=0.4)
            # plt.bar(counts, buildings_ab, color="""randomly pick one""", width=0.4)
            plt.xticks(rotation=75)
            plt.xlabel("Buildings")
            plt.ylabel("No. of instance(s)")
            # plt.xlabel("No. of instance(s)")
            # plt.ylabel("Buildings")
            plt.title("Building Statistics")
            plt.show()
        # elif option == "pie":
        else:
            # throw an exception as there is no other option yet
            raise OptionError

    def show_first(self, option="plain", prioritized_count=4):
        """
        Sort out the statistics of photos, so that the first few given
        count classes that have the most instances will be displayed,
        for the purpose of making training more effective. Return the
        records of these few classes, with names as the keys and counts
        as the values of a mapping type.
        :param option: the option of showing the prioritized statistics.
        :param prioritized_count: the number of buildings prioritized for model training.
        :return: the map of buildings and count of instances.
        """
        stats = self.get_stats()
        sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
        valid_count = 0  # handles the case where Misc comes to a top rank
        index = 0
        first = {}
        if option == "plain":
            print(f"Among the top {prioritized_count} classes:")
            while valid_count < prioritized_count:
                name = sorted_stats[index][0]
                count = sorted_stats[index][1]
                if name != MISC:
                    print(f"{name} has {count} instance(s).")
                    first.update({name: count})
                    valid_count += 1
                index += 1
        elif option == "plot":
            plt.xlabel(f"The most popular {prioritized_count} buildings in the dataset")
            plt.ylabel("No. of instance(s)")
            plt.title(f"Prioritized Building Statistics")
            first_buildings = list(first.keys())
            first_counts = list(first.values())
            plt.plot(first_buildings, first_counts)
        else:
            raise OptionError
        return first

    def extract_frames(self, video, gap=1):
        """
        Extract the frames from a video file, so that
        multiple images could be generated.
        :param video: the video file to extract frames from.

        :return: the list of images that are generated.
        """
        # Read the video from specified path
        cam = cv2.VideoCapture(video)
        try:
            # creating a folder named data
            if not os.path.exists('data'):
                os.makedirs('data')
        # if not created then raise error
        except OSError:
            print('Error: Creating directory of data')
        # frame
        currentframe = 0
        while True:
            # reading from frame
            ret, frame = cam.read()
            if ret:
                # if video is still left continue creating images
                if currentframe % 10 == 0:
                    name = './data/frame' + str(currentframe) + '.jpg'
                    print('Creating...' + name)
                    # writing the extracted images
                    cv2.imwrite(name, frame)
                # increasing counter so that it will
                # show how many frames are created
                currentframe += 1
            else:
                break
        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()

    # rename
    # to compute the total number of buildings that are attached with photos
    # on Google Map, using an API or something like that - ask Jason and Dan
    def record_progress(self, to_worry):
        """
        Record the labelling progress upon the dataset that we have at this moment.
        Display the total amount of classes that the dataset has in total, while showing
        the number of labels that are not in the range of recognition.
        :param to_worry: the number of buildings that the recognizer should be able to identify.
        """
        total = len(self.get_stats().keys())
        no_worry = total - to_worry
        print(f"There are {total} buildings located in the main campus map.\n")
        print(f"Based on the availability of data points on the Google Maps, the number of \n" +
              f"buildings that we would not need to include for training is {no_worry}.\n" +
              f"So we would need to worry about the rest {to_worry}.\n")
        print("There is actually a category called 'Misc' in the training dataset, but " +
              "we have \nexclude this type out from the regular type already. So no worries.\n")
        print(f"Finally, after cutting down the number of buildings to be {to_worry} " +
              "for training, \nour data is ready for the next move.")

    def update_target(self, log, api):
        """
        Update the targeted buildings to recognize, given the
        real-time data available across the repository of Google
        Map.
        :param log: the log of the buildings across the campus.
        :param api: the API of the Google Map to query map data.
        :return: the updated list of buildings to recognize.
        """
        log_path = os.path.join(WEB_ROOT, WEB_REPO[:WEB_REPO.rfind("/")], log)
        # *change this based on the real-time data of Google API:
        # run the notebook, query through API about the photos of buildings,
        # and then automatically download while moving to a corresponding
        # directory on the machine
        worries = []
        no_worries = []
        with open(log_path) as full_log:
            buildings = full_log.readlines()
            to_worry = 0
            no_worry = 0
            total = 0
            for building in buildings:
                building = building.strip()[building.find(" ") + 1:]
                if building not in no_worries:
                    worries.append(building)
                    to_worry += 1
                else:
                    no_worry += 1
                total += 1
        # write a text file that stores the buildings to update
        to_update = os.path.join(WEB_REPO[:WEB_REPO.rfind("/")], "buildings_to_update.txt")
        with open(to_update, "w") as update:
            to_update = [worries[i] + "\n" if i != to_worry - 1 else worries[i] for i in range(to_worry)]
            update.writelines(to_update)

    def split_batch(self, repo):
        """
        Split a repository's photos into batches of 100,
        for a more convenient pre-processing.
        :param repo: the repository to split batches from.
        :return: the new repository that has all the batches split.
        """

    def list_directory(self, directory):
        """
        List out the files in a given directory by performing a
        depth-first traversal. Update the statistics in the file
        manager at the same time.
        :param directory: the path to the directory for returning files.
        :return: the count of files under the given directory.
        """
        instances = os.listdir(directory)
        inst_count = 0
        leaf = directory[directory.rfind("/") + 1:]
        if len(instances) != 0:
            photos = []
            for instance in instances:
                if JPEG not in instance:  # path pointing to a repository
                    inst_count += self.list_directory(os.path.join(directory, instance))
                else:  # path pointing to a photo
                    photos.append(instance)
                    inst_count += 1
            if len(photos) != len(set(photos)):
                missing = self.find_missing(repo=False, batch=directory)
                repeated = self.find_repeated(directory)
                print(f"The repository of {leaf} has {len(missing)} missing and {len(repeated)} repeated data points.")
            if leaf in BUILDINGS:
                self.update_stats(leaf, inst_count)
        else:
            print(f"We do not have any instances for {leaf}.")
        return inst_count

    @staticmethod
    def rename_photo(photo, add_amount):
        """
        Rename a photo in the manner of increasing index, so that the
        format would be corresponding to the sequence of the photo.
        :param photo: the path of the photo to rename.
        :param add_amount: the multiple of MAX_INDEX + 1 to add to the index of the photo.
        :return: a renamed path of the given photo.
        """
        # O(n)
        photo_rep = photo[:photo.find("_") + 1]
        photo_index = photo[photo.find("_") + 1:photo.find(".")]
        photo_format = photo[photo.find("."):]
        old_photo = photo_rep + photo_index + photo_format
        photo_index = int(photo_index) + add_amount * (MAX_INDEX + 1)
        new_photo = photo_rep + str(photo_index) + photo_format
        print(f"{old_photo} has been renamed to {new_photo}.")
        return new_photo

    # sort out the correlation between self and main_repo
    def rename_photos(self, repo):
        """
        Rename the photos inside a repository, so that they would become
        corresponding in terms of ascending order, for differentiation.
        The path of the batch and the index of the repository are given.
        Return the collection of all the renamed photos.
        :param repo: the path of repository that holds all the photos to rename.
        :return: the list of photos that are renamed in the repository.
        """
        # O(n^2)
        batches = [subdir for subdir in os.listdir(repo) if subdir != CACHE]
        renamed = {}
        for batch in batches:
            batch_index = int(batch)
            photos = [photo for photo in os.listdir(batch) if CACHE not in photo]
            for photo in photos:
                old_index = int(photo[photo.rfind("_") + 1:photo.rfind(".")])
                if old_index <= MAX_INDEX:
                    new_index = self.rename_photo(photo, add_amount=batch_index)
                    os.rename(os.path.join(batch, photo), os.path.join(batch, new_index))
                    renamed.update({old_index: new_index})
        if len(renamed) == 0:
            print("Done for renaming photos at this moment.")
        return renamed

    @staticmethod
    def filter_cache(directory):
        """
        Filter out the cache file .DS_Store towards operations on a macOS
        based computer. More to come for the logic towards Windows.
        :param directory: the directory where the cache file exists.
        :return: the list of files in the original directory, except the cache file.
        """
        return [file for file in directory if file != CACHE]

    def merge_photos(self, src, dest):
        """
        Move the photos downloaded from Google Drive, so that they could
        be merged together for error correction or other purposes.
        :param src: the path of the repository that has the sub-directories of photos.
        :param dest: the name of the repository to get merged into.
        :return: the total amount of photos that are merged.
        """
        # O(n^3)
        merged_count = 0
        repo = os.listdir(src)
        dsets = [subdir for subdir in repo if DSET in subdir]
        dset_count = len(dsets)
        merged = dest[dest.rfind("/") + 1:]
        if dset_count == 0:
            print("We can not yet merging any photos.")
        else:
            if merged not in repo:
                os.mkdir(dest)  # assumes that the destination is going to be in repo
            for i in range(dset_count):
                dset = f"{DSET} {i + 1}"
                dset_path = os.path.join(src, dset)
                buildings = self.filter_cache(os.listdir(dset_path))
                for building in buildings:
                    new_repo = os.path.join(dest, building)
                    if building not in os.listdir(dest):
                        os.mkdir(new_repo)
                    building_repo = os.path.join(dset_path, building)
                    photos = self.filter_cache(os.listdir(building_repo))
                    for photo in photos:
                        shutil.move(os.path.join(building_repo, photo), new_repo)
                    merged_count += len(photos)
                print(f"{dset} is completed with moving.")
            else:
                print("Done for merging photos for now.")
        return merged_count

    def find_repeated(self, repo):
        """
        Find the repetitive instance(s) in the repository,
        given the path of the repository. Return the name
        of the building(s) that are in the repository,
        containing these repeated instance(s).
        :param repo: the path of the repository to check.
        :return: the map of building-instances showing repetitions.
        """
        # O(n^2)
        photos = []
        buildings = self.filter_cache(os.listdir(repo))
        repeated = {}
        for building in buildings:
            building_photos = self.filter_cache(os.listdir(os.path.join(repo, building)))
            for instance in building_photos:
                if instance not in photos:
                    photos.append(instance)
                else:
                    if building not in repeated:
                        repeated.update({building: [instance]})
                    else:
                        repeated[building].append(instance)
        if len(repeated) != 0:
            for building, photos in repeated.items():
                print(f"{building} has {photos} repeated.")
                # traversal
        else:
            print(f"There is no repetition at all currently.")
        return repeated

    def find_missing(self, repo, batch):
        """
        Find the missing data points for the repository
        which happens during the labelling process, with
        a given repository that includes incomplete data
        points and a batch for checking the missing points.
        Return the collections of the instances that misses
        from the repository.
        :param repo: repository that includes instances incompletely.
        :param batch: the batch that is complete for checking missing data.
        :return: the collection of missing photos.
        """
        # O(n^2)
        missing = []
        dset = repo[:repo.rfind("/")]
        if batch in os.listdir(dset):
            buildings = self.filter_cache(os.listdir(repo))
            photos = []
            for building in buildings:
                photos += os.listdir(os.path.join(repo, building))
            batch_photos = self.filter_cache(os.listdir(os.path.join(dset, batch)))
            for photo in batch_photos:
                if photo not in photos:
                    missing.append(photo)
            if len(missing) != 0:
                print(f"The photos in the batch missing from the main repository are {missing}.")
            else:
                print("There are not any photos missing from the main repository for now.")
        else:
            print("There is no way for checking missing photos at this moment.")
        return missing
