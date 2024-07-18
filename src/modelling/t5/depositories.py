
import src.functions.directories

class Depositories:

    def __init__(self):
        """
        Constructor
        """

    def exc(self, path):

        directories = src.functions.directories.Directories()
        directories.cleanup(path=path)
        directories.create(path=path)
