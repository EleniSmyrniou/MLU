from tensorflow.keras.callbacks import Callback
import os
import numpy as np
import matplotlib.pyplot as plt


class TestFiguresCallback(Callback):
    """Callback to save test figures after each epoch. """

    def __init__(self, test_input, test_target, checkpoint_path, batch_size, mc_samples=100):
        """ init method
        Args:
            test_input (np.array): test input
            test_target (np.array): test target
            checkpoint_path (str): path to save checkpoints
            mc_samples (int): number of monte carlo samples

        """
        super(TestFiguresCallback, self).__init__()
        self.test_input = test_input
        self.test_target = test_target
        self.checkpoint_path = checkpoint_path
        self.mc_samples = mc_samples
        self.batch_size = batch_size

    def create_test_images(self, epoch: int):
        """ Creates test images.
        Args:
            dest_path (str): local path to save the file
        Returns:
        """

        #plot 2 images
        for i in range(0, len(self.test_input), self.batch_size)[:2]:
            sigmoids = np.zeros((self.mc_samples,) + self.test_target[0].shape)
            for j in range(self.mc_samples):
                # get batch
                batch = self.test_input[i:i + self.batch_size]
                # predict
                batch_sigmoids = self.model.predict(batch, verbose=0)
                # add to sigmoids
                sigmoids[j, :, :, i:i + self.batch_size] = batch_sigmoids[0]
            # Calculates prediction.
            pred_mean = np.mean(sigmoids, axis=0)
            pred_std = np.std(sigmoids, axis=0)
            # plot results with 95% confidence interval
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(self.test_target[i])
            ax[1].imshow(pred_mean[:, :, 0])
            ax[2].imshow(pred_std[:, :, 0])
            # titles
            ax[0].set_title('Target')
            ax[1].set_title('Prediction')
            ax[2].set_title('Uncertainty')
            # add title with the mean pixelwise error
            fig.suptitle('Mean pixelwise error: ' + str(np.mean(np.abs(self.test_target[i] - pred_mean))))
            plt.savefig(os.path.join(self.checkpoint_path, 'test_' + str(i) + "_" + str(epoch) + '.png'))
            plt.close(fig)


    def on_epoch_end(self, epoch, logs=None):
        # every 10th epoch
        if epoch % 10 == 0:
            self.create_test_images(epoch)