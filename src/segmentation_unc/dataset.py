import os
import numpy as np
import PIL.Image as Image

def get_train_test_validation_data(directory, test, validation):
    # get all figures from directory
    figures = os.listdir(directory)
    input = [figure for figure in figures if 'noisy' in figure]
    target = [figure for figure in figures if 'noisy' not in figure]
    # split into train, test, validation
    train_input = input[:int(len(input) * (1 - test - validation))]
    train_target = target[:int(len(target) * (1 - test - validation))]
    test_input = input[int(len(input) * (1 - test - validation)):int(len(input) * (1 - validation))]
    test_target = target[int(len(target) * (1 - test - validation)):int(len(target) * (1 - validation))]
    validation_input = input[int(len(input) * (1 - validation)):]
    validation_target = target[int(len(target) * (1 - validation)):]
    # load data from png files
    train_input = np.array([np.array(Image.open(os.path.join(directory, figure))) for figure in train_input])
    train_target = np.array([np.array(Image.open(os.path.join(directory, figure))) for figure in train_target])
    test_input = np.array([np.array(Image.open(os.path.join(directory, figure))) for figure in test_input])
    test_target = np.array([np.array(Image.open(os.path.join(directory, figure))) for figure in test_target])
    validation_input = np.array([np.array(Image.open(os.path.join(directory, figure))) for figure in validation_input])
    validation_target = np.array([np.array(Image.open(os.path.join(directory, figure))) for figure in validation_target])
    # normalize data
    train_input = train_input / 255
    train_target = train_target / 255
    test_input = test_input / 255
    test_target = test_target / 255
    validation_input = validation_input / 255
    validation_target = validation_target / 255
    return train_input, train_target, test_input, test_target, validation_input, validation_target