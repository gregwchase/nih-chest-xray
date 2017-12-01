import cv2
import os
import time


def create_directory(directory):
    """
    Creates a new folder in the specified directory if folder doesn't exist.

    INPUT
        directory: Folder to be created, called as "folder/".

    OUTPUT
        New folder in the current directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def crop_and_resize_images(path, new_path, img_size):
    """
    Crops, resizes, and stores all images from a directory in a new directory.

    INPUT
        path: Path where the current, unscaled images are contained.
        new_path: Path to save the resized images.
        img_size: New size for the rescaled images.

    OUTPUT
        All images cropped, resized, and saved to the new folder.
    """
    create_directory(new_path)
    dirs = [l for l in os.listdir(path) if l != '.DS_Store']
    # total = 0

    for item in dirs:
        # Read in all images as grayscale
        img = cv2.imread(path + item, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        cv2.imwrite(str(new_path + item), img)
        # total += 1
        # print("Saving: ", item, total)


if __name__ == '__main__':
    start_time = time.time()
    crop_and_resize_images(path='../data/images/', new_path='../data/resized-256/', img_size=256)
    print("Seconds: ", time.time() - start_time)
