import os
from datetime import datetime, timedelta
from shutil import rmtree


def generate_formatted_directory_path(interval=1):
    today = datetime.utcnow()
    delta = timedelta(hours=interval)
    delete_date = today - delta
    formatted_date = delete_date.strftime('%Y%m%d%H%M%S')[:-4]
    return formatted_date


def delete_directory(directory_name):
    # If file exists, delete it
    if os.path.exists(directory_name):
        rmtree(directory_name)


def delete_file(file_name):
    # If file exists, delete it
    if os.path.isfile(file_name):
        os.remove(file_name)
    else:  # Show an error
        print("Error: %s file not found" % file_name)


delete_directory(generate_formatted_directory_path())



