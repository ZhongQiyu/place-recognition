"""
A Python class that parses external image data, so that they
could get together with the help of
"""
import cv2


class DataParser():

    def transform(self, image, format="jpeg"):
        """
        Transform a given image from the HEIC raw format into
        a given format, so that the data would be more readable
        by computer vision modules.
        """
        data = cv2.imread(image)
