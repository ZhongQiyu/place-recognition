# consider inheritance functions from original Python modules
class ImageStack():
    """
    Implement an image stack to push the array of the images in the repo
    onto the processing pipeline of the recognizer. This stack is designed
    to lessen the RAM's work, when assign the processed images to the pipeline,
    during the training process of the CNN-based models.
    """

    # https://realpython.com/how-to-implement-python-stack/

    def __init__(self, data):
        self.data = data
