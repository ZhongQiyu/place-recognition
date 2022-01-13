class Preprocessor():

    def __init__(self):
        """
        Initialize a pre-processor for the data to train model with.
        """

        self.method = ""
        self.preprocessed = ""

    # *design a HOG_SVM module and put into recognizer
    def resize_all(src, pklname, include, width=150, height=None):
        """
        Load images from path, resize them and write them as arrays to a dictionary,
        together with labels and metadata. The dictionary is written to a pickle file
        named '{pklname}_{width}x{height}px.pkl'.
        :param src: path to data
        :param pklname: path to output file
        :param width: target width of the image in pixels
        :include: classes of images for resizing
        """
        super_directory = src[:src.rfind("/")]
        pklname = f"{pklname}_{width}x{height}px.pkl"
        if os.path.exists(os.path.join(super_directory, pklname)):
            print("The pickle package has already exists.")
        else:
            height = height if height is not None else width
            data = dict()
            data['description'] = 'resized ({0}x{1})Union College Building images in rgb'.format(int(width),
                                                                                                 int(height))
            data['label'] = []
            data['filename'] = []
            data['data'] = []
            # read all images in PATH, resize and write to DESTINATION_PATH
            for subdir in os.listdir(src):
                if subdir in include:
                    print(f"Resizing the instances in {subdir}..")
                    current_path = os.path.join(src, subdir)
                    for file in os.listdir(current_path):
                        if file[-3:] in {'jpg', 'png'} or file[-4:] in {"jpeg"}:
                            im = imread(os.path.join(current_path, file))
                            im = resize(im, (width, height))  # [:,:,::-1]
                            data['label'].append(subdir)
                            data['filename'].append(file)
                            data['data'].append(im)
                joblib.dump(data, pklname)
            shutil.move(pklname, super_directory)