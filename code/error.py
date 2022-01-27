# enable throwing exceptions for missing the options
class OptionError(Exception):
    """
    Give an option to indicate the error upon the selection of options
    across the module of the PhotoManager. Provide external and extra
    reminders to the developer, which is me, to code in a better way,
    and this also provided some robustness towards the real program.
    """

    def __init__(self, message, errors):
        """
        Write out the error message, by highlighting the type of
        errors.
        :param message: the error message to write.
        :param errors: the number of errors that brings up in the program.
        """
        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.errors = errors