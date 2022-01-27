import shutil
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from constants import Constants
from error import OptionError


# consider inheritance from an original Python library
class PhotoManager():
    """
    A photo manager that handles all the available photos as raw files,
    ensuring the pipeline for training the recognizer would work.
    """
    def __init__(self, mode="web", stats={}):
        """
        Initialize a photo manager for the photo repository,
        so that the data would be ready for training. This
        initializer allows passing in a map of statistics
        over the photos.
        :param mode: the mode to initialize the photo manager.
        :param stats: the statistics to store the data in the photo manager.
        """
        self.constants = Constants(mode=mode)
        self.mode = mode
        self.stats = stats

    def get_constants(self):
        """
        Return the constant-based values for the photo manager.
        """
        return self.constants

    def set_constants(self, new_mode):
        """
        Set the constants with a new mode, that changes the
        configuration of the photo manager.
        If the new mode is the same as the old one, the parameters
        would be reset to the default values. Otherwise, the other
        set of parameters would be used for powering the manager.
        :param new_mode: the new mode for configuration.
        """
        self.constants = Constants(new_mode)

    def get_mode(self):
        """
        Return the mode of the photo manager.
        """
        return self.mode

    def set_mode(self, new_mode):
        """
        Set the mode of the photo manager with a new one,
        should be either the mode of web or local repository.
        :param new_mode: the new mode to set upon the photo manager.
        """
        self.mode = new_mode
        self.set_constants(new_mode)

    def get_stats(self):
        """
        Return the statistics of the photos stored in the recognizer.
        """
        return self.stats

    def flatten(self, directory):
        """
        Return a flattened version of an arbitrary directory.
        :param directory: the directory to flatten.
        :return: the flattened collection of the original directory.
        """
        if len(directory) == 0:
            return directory
        if isinstance(directory[0], list):
            return self.flatten(directory[0]) + self.flatten(directory[1:])
        return directory[:1] + self.flatten(directory[1:])

    def set_stats(self, data_path):
        """
        Set the statistics of the instances stored in the recognizer,
        with any new form of the data.
        :param data_path: the path of the repository where the instances are placed in.
        """
        jpeg = self.get_constants().JPEG
        buildings = [directory for directory in self.filter_cache(os.listdir(data_path))]
        data = {}
        for building in buildings:
            building_path = os.path.join(data_path, building)
            building_data = self.filter_cache(os.listdir(building_path))
            for inst in building_data:
                if jpeg not in inst:  # it is a collection of extracted frames
                    inst_index = building_data.index(inst)
                    building_data[inst_index] = self.filter_cache(os.listdir(os.path.join(building_path, inst)))
            data.update({building: len(self.flatten(building_data))})
        self.stats = data

    def check_missing(self):
        """

        """

    def check_repeated(self):
        """

        """

    def show_stats(self, option="plain"):
        """
        Display the statistics of photos in a more formatted way,
        with the help of the attribute itself.
        :param option: plain for plain text, plot for matplotlib.pyplot.plot()
        """
        inst_bound = self.constants.BOUND
        stats = self.get_stats()
        filtered_stats = {name: count for name, count in stats.items() if count >= inst_bound}
        if option == "plain":  # plain text
            if len(filtered_stats) != 0:
                for name, count in filtered_stats.items():
                    print(f"There are {count} instance(s) for {name}.")
                print(f"There are {sum(list(filtered_stats.values()))} instance(s) in total.")
            else:
                print("Currently, there is no data for the statistics.")
        elif option == "table" or option == "plot":
            buildings = [building for building in list(self.get_stats().keys()) if building in filtered_stats.keys()]
            counts = [filtered_stats[building] for building in buildings]
            if option == "table":
                table_stats = pd.DataFrame()
                table_stats["Building"] = buildings
                table_stats["Count"] = counts
                print(tabulate(table_stats, headers="keys", tablefmt="psql"))
            else:
                # bar plot
                buildings_pos = [pos for pos in range(len(buildings))]
                buildings_ab = tuple(["".join([char for char in building if char.isupper()]) for building in buildings])
                plt.barh(buildings_pos, counts, color='maroon', tick_label=buildings_ab)
                plt.yticks()
                plt.xlabel("No. of instance(s)")
                plt.ylabel("Buildings")
                plt.title('Building Distribution From Data')
                plt.show()
        else:  # throw an exception as there is no other option yet
            raise OptionError
        print(f"There are {sum(list(stats.values()))} instance(s) in total.")

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
        misc = Constants().MISC
        stats = self.get_stats()
        sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
        valid_count = 0  # handles the case where Misc comes to a top rank
        index = 0
        first = {}
        while valid_count < prioritized_count:
            name = sorted_stats[index][0]
            count = sorted_stats[index][1]
            if name != misc:
                first.update({name: count})
                valid_count += 1
            index += 1
        if option == "plain":
            print(f"Among the top {prioritized_count} classes:")
            for name, count in first.items():
                print(f"{name} has {count} instance(s).")
        elif option == "plot" or option == "pie":
            first_buildings = list(first.keys())
            first_counts = list(first.values())
            if option == "plot":
                plt.xlabel(f"The most popular {prioritized_count} buildings in the dataset")
                plt.ylabel("No. of instance(s)")
                plt.title(f"Prioritized Building Statistics")
                plt.plot(first_buildings, first_counts)
            else:  # elif option == "pie":
                sizes = [float(str(count/sum(first_counts)).format(".2f")) for count in first_counts]
                explode = [0 if size < max(sizes) else 0.1 for size in sizes]
                fig1, ax1 = plt.subplots()
                ax1.pie(sizes, explode=explode, labels=first_buildings, autopct='%1.1f%%', shadow=True, startangle=90)
                ax1.axis('equal')
            plt.show()
        else:
            raise OptionError
        return first

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

    def rename(self, inst, coef):
        """
        Rename an instance in the manner of increasing index, so that
        the format would be corresponding to the sequence of the photo.
        :param inst: the path of the instance to rename.
        :param coef: the multiple of MAX_INDEX + 1 to add to the index of the photo.
        :return: a renamed path of the given photo.
        """
        # O(n)
        max_index = self.get_constants().MAX_INDEX
        rep = inst[:inst.find("_") + 1]
        index = inst[inst.find("_") + 1:inst.find(".")]
        fmt = inst[inst.find("."):]
        index = int(index) + coef * (max_index + 1)
        new_inst = rep + str(index) + fmt
        return new_inst

    def filter_cache(self, directory):
        """
        Filter out the cache file .DS_Store towards operations on a macOS
        based computer. More to come for the logic towards Windows.
        :param directory: the directory where the cache file exists.
        :return: the list of files in the original directory, except the cache file.
        """
        cache = self.get_constants().CACHE
        return [file for file in directory if file != cache]

    def rename_all(self, repo, coef=1):
        """
        Rename the instances inside a repository, so that the main repository
        would become corresponding in index, in terms of ascending order.
        The path of the repository and the default amount to add is given.
        Return the collection of all the renamed photos.
        :param repo: the path of repository that holds all the photos to rename.
        :param coef: the coefficient to add upon the current indices of photos.
        :return: the map of old and new indices for the photos in the repository.
        """
        # O(n^2)
        max_index = self.get_constants().MAX_INDEX
        insts = self.filter_cache(os.listdir(repo))
        renamed = []
        for inst in insts:
            index = int(inst[inst.rfind("_") + 1:inst.rfind(".")])
            fmt = inst[inst.rfind(".")+1:]
            if index < (max_index + 1) * coef:
                new_inst = self.rename(inst, coef=coef)
                old_path = os.path.join(repo, inst)
                new_path = os.path.join(repo, new_inst)
                os.rename(old_path, new_path)
                new_index = int(new_inst[new_inst.rfind("_")+1:new_inst.rfind(".")])
                renamed.append((index, new_index, fmt))
        if len(renamed) == 0:
            print("Done for renaming at this moment.")
        else:
            print(f"{len(renamed)} instance(s) has been renamed.")
        return renamed

    def imagify(self, video, gap=5):
        """
        Extract the frames from a video file, so that
        multiple images could be generated tagged to
        the video.
        :param video: the video file to extract frames from.
        :param gap: the gap of frames for extraction.
        :return: the list of images that are generated.
        """
        jpeg = self.get_constants().JPEG
        mov = self.get_constants().MOV
        video_root = video[:video.rfind("/")]
        video_name = video[video.rfind("/")+1:video.rfind(".")]
        video_dir = os.path.join(video_root, video_name)
        cam = cv2.VideoCapture(video)
        try:
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
        except OSError:
            print(f'Error: Creating directory of {video_name}')
        currentframe = 0
        extracted = []
        while True:
            ret, frame = cam.read()
            if ret:
                if currentframe % gap == 0:
                    name = f"{video_name}_{currentframe // gap}{jpeg}"
                    os.chdir(video_dir)
                    cv2.imwrite(name, frame)
                    extracted.append(name)
                currentframe += 1
            else:
                break
        print(f"{video_name + mov} is just compiled to images.")
        cam.release()
        cv2.destroyAllWindows()
        return extracted

    def list_directory(self, directory):
        """
        List out the files in a given directory by performing a
        depth-first traversal. Update the statistics in the file
        manager at the same time.
        :param directory: the path to the directory for returning files.
        :return: the count of files under the given directory.
        """
        jpeg = self.get_constants().JPEG
        buildings = self.get_constants().BUILDINGS
        instances = os.listdir(directory)
        inst_count = 0
        leaf = directory[directory.rfind("/") + 1:]
        if len(instances) != 0:
            photos = []
            for instance in instances:
                if jpeg not in instance:  # path pointing to a repository
                    inst_count += self.list_directory(os.path.join(directory, instance))
                else:  # path pointing to a photo
                    photos.append(instance)
                    inst_count += 1
            if len(photos) != len(set(photos)):
                # missing = self.find_missing(repo=directory, batch=directory)
                repeated = self.find_repeated(directory)
                # print(f"The repository of {leaf} has {len(missing)} missing and {len(repeated)} repeated data points.")
            if leaf in buildings:
                self.update_stats(leaf, inst_count)
        else:
            print(f"We do not have any instances for {leaf}.")
        return inst_count

    def simple_merge(self, src, dest):
        """
        Merge the repositories from the source to the destination,
        in a relatively simple manner, i.e., no concatenated directories
        from the main repository.
        :param src: the source of the dataset, to get the images out for merging.
        :param dest: the destination of the repository to put the merged data in.
        :return: the count of instance(s) that is gotten merged.
        """
        dset_count = len(src)
        merged_count = 0
        for i in range(dset_count):
            subdset = self.filter_cache(os.listdir(src[i]))
            for inst in subdset:
                shutil.move(os.path.join(src[i], inst), dest)
                merged_count += 1
        return merged_count

    def subdir_merge(self, src, dest):
        """
        Merge the subdirectories together, so that the stored instances would
        come together in a main directory. This way of merging files would be
        relative to merging simply, as there are layers of merging in this one
        instead of a simple layer of instances. Return the total count of the
        merged instances.
        :param src: the source of the data, where all the subdirectories are stored.
        :param dest: the destination to put all the instances together.
        :return: the total amount of instances that are merged together.
        """
        dset_count = len(src)
        merged_count = 0
        for i in range(dset_count):
            subdset = src[i]
            buildings = self.filter_cache(os.listdir(subdset))
            for building in buildings:
                new_repo = os.path.join(dest, building)
                if building not in os.listdir(dest):
                    os.mkdir(new_repo)
                building_repo = os.path.join(subdset, building)
                photos = self.filter_cache(os.listdir(building_repo))
                for photo in photos:
                    shutil.move(os.path.join(building_repo, photo), new_repo)
                merged_count += len(photos)
            print(f"Subdirectory {subdset} is completed with moving.")
        else:
            print("Done for merging photos for now.")
        return merged_count

    def merge(self, src, dest):
        """
        Move the photos downloaded from Google Drive, so that they could
        be merged together for error correction or other purposes.
        :param src: the collection of the subdirectories of photos.
        :param dest: the name of the repository to get merged into.
        :return: the total amount of photos that are merged.
        """
        # O(n^3)
        img = self.get_constants().IMG
        merged_count = 0
        dset_count = len(src)
        if dset_count == 0:
            print("We can not yet merging any photos.")
        else:
            if not os.path.exists(dest):
                os.mkdir(dest)
            mode = "simple_comp" if all([img in file for file in os.listdir(src[0])]) else "subdir_comp"
            print(mode)
            merged_count = self.subdir_merge(src, dest) if mode == "subdir_comp" else self.simple_merge(src, dest)
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

# ----NOT YET IMPORTANT----

    # to modify
    def update_target(self, log, api):
        """
        Update the targeted buildings to recognize, given the
        real-time data available across the repository of Google
        Map.
        :param log: the log of the buildings across the campus.
        :param api: the API of the Google Map to query map data.
        :return: the updated list of buildings to recognize.
        """
        root = self.get_constants().ROOT
        repo = self.get_constants().REPO
        log_path = os.path.join(root, repo, log)
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
        to_update = os.path.join(root, repo, "buildings_to_update.txt")
        with open(to_update, "w") as update:
            to_update = [worries[i] + "\n" if i != to_worry - 1 else worries[i] for i in range(to_worry)]
            update.writelines(to_update)
