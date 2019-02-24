import os
import glob
import argparse
import numpy as np
from shutil import copyfile


def create_train_val_split(data_path: str, data_path_train: str, val_ratio: float = 0.2) -> None:
    """
    Creates from an original "train" folder with the following structure:
        - train/
            category_1/
                image_1.png
                image_2.png
            category_2/
                image_1.png
                image_2.png
            ...

    :param data_path: path where the data of the project is located
    :param data_path_train:  path where the train data of the project is located
    :param val_ratio: ration of validation, in respect to train size
    :return: None, copies the files with the same file structure under "train_split" and "val_split",
            maintaining the ratio selected.
    """

    categories = glob.glob(data_path_train+'/**', recursive=False)
    categories = [element.replace(data_path_train, '') for element in categories]

    os.mkdir(data_path+'train_split')
    os.mkdir(data_path+'val_split')

    for category in categories:
        cat_images = glob.glob(data_path_train + category + '/*.png', recursive=False)
        val_cat_images = list(np.random.choice(cat_images,
                                               size=int(round(len(cat_images) * val_ratio, 0)), replace=False))
        train_cat_images = [element for element in cat_images if element not in val_cat_images]

        os.mkdir(data_path + 'train_split/' + category)
        os.mkdir(data_path + 'val_split/' + category)

        for train_cat_image in train_cat_images:
            copyfile(train_cat_image, train_cat_image.replace(data_path_train, data_path + 'train_split/'))

        for val_cat_image in val_cat_images:
            copyfile(val_cat_image, val_cat_image.replace(data_path_train, data_path + 'val_split/'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='data/', help='data path')
    parser.add_argument('--data_train', '-dt', type=str, default='data/train/', help='data train path')
    parser.add_argument('--percentage_val', '-pv', type=float, default=0.2, help='validation ratio per class')
    args = parser.parse_args()

    data_path = args.data
    data_path_train = args.data_train
    val_percent = args.percentage_val

    create_train_val_split(data_path=data_path, data_path_train=data_path_train, val_ratio=val_percent)
    print('Process finished properly!')
