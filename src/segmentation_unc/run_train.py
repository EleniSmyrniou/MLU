from train import train
# mute warnings
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    epochs = 50
    batch_size = 1
    initial_epoch = 0
    #directory = "D:\MLU\data\synthetic_boreholes"
    directory = "D:\MLU\data\jsons_reduced"
    test_percentage = 0.1
    validation_percentage = 0.1
    train(epochs, batch_size, initial_epoch, directory, test_percentage, validation_percentage)
