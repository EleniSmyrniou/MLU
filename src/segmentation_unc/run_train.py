from train import train

if __name__ == "__main__":
    epochs = 1
    batch_size = 1
    initial_epoch = 0
    directory = "D:\MLU\data\sub_dir"
    test_percentage = 0.1
    validation_percentage = 0.1
    train(epochs, batch_size, initial_epoch, directory, test_percentage, validation_percentage)
