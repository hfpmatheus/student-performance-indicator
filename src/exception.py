import sys

def format_error_message(error_message, error_detail: sys):
    '''
    The purpose of this function is to extract information about the error, such as the file name, 
    line number, and error message, and return a formatted error message string.
    '''
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_msg = str(error_message)

    formatted_error_message = f"Error ocurred in python script named [{file_name}], line number [{line_number}].\n Error mesage: [{error_msg}]"

    return formatted_error_message

# This line defines a custom exception class CustomException that inherits from the built-in Exception class.
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super.__init__(error_message)
        self.error_message = format_error_message(error=error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message