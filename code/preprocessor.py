import os
import joblib
import shutil
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
from photo_manager import PhotoManager, Constants


"""
A Python class that simulates the pre-processing part of a machine learning
project, in this case specifically the pre-processing towards image-based data.
"""


class Preprocessor():

    def __init__(self):
        """
        Initialize a pre-processor for the data to train model with.
        """
        self.constants = Constants()
        self.method = ""
        self.preprocessed = ""

    def get_constants(self):
        """
        :return: the constants to use during the preprocessing part.
        """
        return self.constants

    def get_method(self):
        """
        :return: the method of carrying out the processing.
        """
        return self.method

    def get_preprocessed(self):
        """
        :return: the report of the pre-processing module after it runs.
        """
        return self.preprocessed

    # *design a HOG_SVM module and put into recognizer
    def resize_all(self, src, pklname, include, width=150, height=None):
        """
        Load images from path, resize them and write them as arrays to a dictionary,
        together with labels and metadata. The dictionary is written to a pickle file
        named '{pklname}_{width}x{height}px.pkl'.
        :param src: path to data
        :param pklname: path to output file
        :param width: target width of the image in pixels
        :include: classes of images for resizing
        """
        const = self.get_constants()
        super_directory = src[:src.rfind("/")]
        pklname = f"{pklname}_{width}x{height}px.pkl"
        if os.path.exists(os.path.join(super_directory, pklname)):
            print("The pickle package has already exists.")
        else:
            height = height if height is not None else width
            data = dict()
            data['description'] = 'resized ({0}x{1})Union College building RGB images'.format(int(width), int(height))
            data['label'] = []
            data['filename'] = []
            data['data'] = []
            for subdir in os.listdir(src):
                if subdir in include:
                    print(f"Resizing the instances in {subdir}..")
                    current_path = os.path.join(src, subdir)
                    for file in os.listdir(current_path):
                        file_format = file[file.find(".")-1:]
                        if any([file_format in {const.JPG, const.PNG, const.JPEG}]):
                            im = imread(os.path.join(current_path, file))
                            im = resize(im, (width, height))
                            data['label'].append(subdir)
                            data['filename'].append(file)
                            data['data'].append(im)
                joblib.dump(data, pklname)
            shutil.move(pklname, super_directory)
            print("The resizing operations have completed.")

    def plot_bar(self, y, loc='left', relative=True):  # generalize
        """
        Plot the distribution of the signals in the training data,
        given the signals, the location to put the legend, and a
        relation of the location with the whole canvas.
        :param y: the signals of the training data.
        :param loc: the location to place the legend.
        :param relative: the relation of the legend with the whole canvas.
        """
        width = 0.35
        if loc == 'left':
            n = -0.5
        else:  # elif loc == 'right':
            n = 0.5
        # calculate counts per type and sort, to ensure their order
        unique, counts = np.unique(y, return_counts=True)
        sorted_index = np.argsort(unique)
        unique = unique[sorted_index]
        if relative:  # plot as a percentage
            counts = 100 * counts[sorted_index] / len(y)
            ylabel_text = '% count'
        else:  # plot counts
            counts = counts[sorted_index]
            ylabel_text = 'count'
        xtemp = np.arange(len(unique))
        plt.bar(xtemp + n * width, counts, align='center', alpha=.7, width=width)
        plt.xticks(xtemp, unique, rotation=45)
        plt.xlabel('building name')
        plt.ylabel(ylabel_text)

    def data_info(self, base_name, data_path, height, width):
        """
        Display the information of the data that is prepared
        for the model training and testing process.
        :param base_name: the name of the directory to handle the data.
        :param data_path: the path of the data where stuff are placed.
        :param height: the universal height of the images.
        :param width: the universal width of the images.
        :return: the data of the images.
        """
        resized = data_path[:data_path.rfind("/")]
        data = joblib.load(os.path.join(resized, f'{base_name}_{width}x{height}px.pkl'))
        print('number of samples: ', len(data['data']))
        print('keys: ', list(data.keys()))
        print('description: ', data['description'])
        print('image shape: ', data['data'][0].shape)
        print('labels:', np.unique(data['label']))
        print(Counter(data['label']))
        return data

    def plot_classes(self, data):
        """
        Plot the classes of the buildings to predict instance(s) with.
        :param data: the original dataset where the images of the buildings are stored.
        """
        # use np.unique to get all unique values in the list of labels
        labels = np.unique(data['label'])
        # set up the matplotlib figure and axes, based on the number of labels
        fig, axes = plt.subplots(1, len(labels))
        fig.set_size_inches(10, 4)  # magic numbers
        fig.tight_layout()
        # make a plot for every label (equipment) type. The index method returns the
        # index of the first item corresponding to its search string, label in this case
        for ax, label in zip(axes, labels):
            idx = data['label'].index(label)
            ax.imshow(data['data'][idx])
            ax.axis('off')
            ax.set_title(label)

    def plot_instances(self, data):
        """
        Plot the distribution of the dataset using bar chart.
        :param data: the dataset to use for model training and testing.
        """
        # make X and y
        X = np.array(data['data'])
        y = np.array(data['label'])
        # cross-validate
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
        # plot the distribution of the data
        plt.suptitle('relative amount of photos per building')
        self.plot_bar(y_train, loc='left')
        self.plot_bar(y_test, loc='right')
        plt.legend(['train ({0} photos)'.format(len(y_train)), 'test ({0} photos)'.format(len(y_test))])

    def preprocess(self, best_sensed, base_name, compression_ratio):
        """
        Make the pipeline of the whole pre-processing module, and
        set up the image data ready for more processing i.e. model
        training and testing through the whole project. Pass in the
        names of buildings that make the best sense, the name to put
        the images, and the ratio of compressing the image data.
        :param best_sensed: the names of buildings to include for classification.
        :param base_name: the name of the Pickle repository to store images in.
        :param compression_ratio: the ratio of compressing the images.
        """
        # set up the data path
        photo_manager = PhotoManager()
        const = photo_manager.get_constants()
        repo = photo_manager.get_constants().REPO
        dset = photo_manager.get_constants().DSET
        data_path = os.path.join(repo, dset)
        width, height = (const.IMG_LENGTH * compression_ratio, const.IMG_WIDTH * compression_ratio)
        include = list(best_sensed.keys())
        # resize the data, and display the information
        self.resize_all(src=data_path, pklname=base_name, include=include, width=width, height=height)
        print()
        data = self.data_info(base_name, data_path, height, width)
        print()
        self.plot_classes(data)
        print()
        self.plot_instances(data)
