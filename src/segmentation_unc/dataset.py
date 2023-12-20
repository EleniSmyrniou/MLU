import os
import numpy as np
import PIL.Image as Image


def get_train_test_validation_data_for_Simple_mplr(directory, test, validation, normalize=True):
    # get all figures from directory
    figures = os.listdir(directory)
    input = [figure for figure in figures if 'noisy' in figure and 'test' not in figure]
    target = [figure for figure in figures if 'noisy' not in figure and 'test' not in figure]
    # split into train, test, validation
    train_input = input[:int(len(input) * (1 - test - validation))]
    train_target = target[:int(len(target) * (1 - test - validation))]
    test_input = input[int(len(input) * (1 - test - validation)):int(len(input) * (1 - validation))]
    test_target = target[int(len(target) * (1 - test - validation)):int(len(target) * (1 - validation))]
    validation_input = input[int(len(input) * (1 - validation)):]
    validation_target = target[int(len(target) * (1 - validation)):]
    # load data from npy files
    train_input = np.array([np.load(os.path.join(directory, figure)) for figure in train_input])
    train_target = np.array([np.load(os.path.join(directory, figure)) for figure in train_target])
    test_input = np.array([np.load(os.path.join(directory, figure)) for figure in test_input])
    test_target = np.array([np.load(os.path.join(directory, figure)) for figure in test_target])
    validation_input = np.array([np.load(os.path.join(directory, figure)) for figure in validation_input])
    validation_target = np.array([np.load(os.path.join(directory, figure)) for figure in validation_target])
    # flatten input and target so that they are arrays of shape ( None, 1)
    train_input = np.reshape(train_input, (train_input.shape[0] * train_input.shape[1], 1))
    train_target = np.reshape(train_target, (train_target.shape[0] * train_target.shape[1], 1))
    test_input = np.reshape(test_input, (test_input.shape[0] * test_input.shape[1], 1))
    test_target = np.reshape(test_target, (test_target.shape[0] * test_target.shape[1], 1))
    validation_input = np.reshape(validation_input, (validation_input.shape[0] * validation_input.shape[1], 1))
    validation_target = np.reshape(validation_target, (validation_target.shape[0] * validation_target.shape[1], 1))


    # change data type of target to int
    train_target = train_target.astype(int)
    test_target = test_target.astype(int)
    validation_target = validation_target.astype(int)
    if normalize:
        # normalize input
        train_input = train_input / max(train_input.max(), test_input.max(), validation_input.max())
        test_input = test_input / max(train_input.max(), test_input.max(), validation_input.max())
        validation_input = validation_input / max(train_input.max(), test_input.max(), validation_input.max())

    return train_input, train_target, test_input, test_target, validation_input, validation_target

def get_train_test_validation_data_jsons(directory, test, validation, normalize=True, reshape=True):
    # get all figures from directory
    figures = os.listdir(directory)
    all_data = [figure for figure in figures if 'pairs' in figure and 'test' not in figure]
    # split into train, test, validation
    train_data = all_data[:int(len(all_data) * (1 - test - validation))]
    test_data = all_data[int(len(all_data) * (1 - test - validation)):int(len(all_data) * (1 - validation))]
    validation_data = all_data[int(len(all_data) * (1 - validation)):]
    # load data from npy files
    train_input = np.array([np.load(os.path.join(directory, figure))[0] for figure in train_data])
    train_target = np.array([np.load(os.path.join(directory, figure))[1] for figure in train_data])
    test_input = np.array([np.load(os.path.join(directory, figure))[0] for figure in test_data])
    test_target = np.array([np.load(os.path.join(directory, figure))[1] for figure in test_data])
    validation_input = np.array([np.load(os.path.join(directory, figure))[0] for figure in validation_data])
    validation_target = np.array([np.load(os.path.join(directory, figure))[1] for figure in validation_data])
    # size of input is (None, 64 , 1) -> (None, 64, 64, 1) by adding a dimension and repeating the values
    if reshape:
        train_input = np.repeat(train_input, 64, axis=2)
        train_input = np.expand_dims(train_input, axis=3)
        test_input = np.repeat(test_input, 64, axis=2)
        test_input = np.expand_dims(test_input, axis=3)
        validation_input = np.repeat(validation_input, 64, axis=2)
        validation_input = np.expand_dims(validation_input, axis=3)
        # size of target is (None, 64, 64) -> (None, 64, 64, 1) by adding a dimension and repeating the values
        train_target = np.repeat(train_target, 64, axis=2)
        train_target = np.expand_dims(train_target, axis=3)
        test_target = np.repeat(test_target, 64, axis=2)
        test_target = np.expand_dims(test_target, axis=3)
        validation_target = np.repeat(validation_target, 64, axis=2)
        validation_target = np.expand_dims(validation_target, axis=3)
    # change data type of target to int
    train_target = train_target.astype(int)
    test_target = test_target.astype(int)
    validation_target = validation_target.astype(int)
    if normalize:
        # normalize target
        train_target = train_target / max(train_target.max(), test_target.max(), validation_target.max())
        test_target = test_target / max(train_target.max(), test_target.max(), validation_target.max())
        validation_target = validation_target / max(train_target.max(), test_target.max(), validation_target.max())
        # normalize input
        train_input = train_input / max(train_input.max(), test_input.max(), validation_input.max())
        test_input = test_input / max(train_input.max(), test_input.max(), validation_input.max())
        validation_input = validation_input / max(train_input.max(), test_input.max(), validation_input.max())

    return train_input, train_target, test_input, test_target, validation_input, validation_target


def get_train_test_validation_data(directory, test, validation):
    # get all figures from directory
    figures = os.listdir(directory)
    input = [figure for figure in figures if 'noisy' in figure and 'test' not in figure]
    target = [figure for figure in figures if 'noisy' not in figure and 'test' not in figure]
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
    # crop images to 945x26 pixels
    train_input = train_input[:, 0:945, 495:508, :3]
    train_target = train_target[:, 0:945, 495:508, :3]
    test_input = test_input[:, 0:945, 495:508, :3]
    test_target = test_target[:, 0:945, 495:508, :3]
    validation_input = validation_input[:, 0:945, 495:508, :3]
    validation_target = validation_target[:, 0:945, 495:508, :3]
    # resize images to 256x256 pixels
    pixels = 64
    train_input = np.array([np.array(Image.fromarray(figure).resize((pixels, pixels))) for figure in train_input])
    train_target = np.array([np.array(Image.fromarray(figure).resize((pixels, pixels))) for figure in train_target])
    test_input = np.array([np.array(Image.fromarray(figure).resize((pixels, pixels))) for figure in test_input])
    test_target = np.array([np.array(Image.fromarray(figure).resize((pixels, pixels))) for figure in test_target])
    validation_input = np.array([np.array(Image.fromarray(figure).resize((pixels, pixels))) for figure in validation_input])
    validation_target = np.array([np.array(Image.fromarray(figure).resize((pixels, pixels))) for figure in validation_target])
    # change data type of target to int
    train_target = train_target.astype(int)
    test_target = test_target.astype(int)
    validation_target = validation_target.astype(int)
    # normalize data
    train_input = train_input / 255
    train_target = train_target
    test_input = test_input / 255
    test_target = test_target
    validation_input = validation_input / 255
    validation_target = validation_target

    return train_input, train_target, test_input, test_target, validation_input, validation_target