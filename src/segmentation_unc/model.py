from bayesian_unet import bayesian_unet
import os
import datetime

def get_model(input_shape, kernel_size=3, activation="relu", padding="SAME", prior_std=1):
    # check point with time stamp
    checkpoint_path = ("/bayesian/bayesian" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-{epoch:02d}")
    net = bayesian_unet
    model = net(input_shape,
                kernel_size=kernel_size,
                activation=activation,
                padding=padding,
                prior_std=prior_std)
    model.summary(line_length=127)
    loss = "KLDivergence"
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model, checkpoint_path