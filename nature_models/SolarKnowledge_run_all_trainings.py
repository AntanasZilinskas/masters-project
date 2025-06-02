'''
 (c) Copyright 2023
 All rights reserved
 Programs written by Yasser Abduallah
 Department of Computer Science
 New Jersey Institute of Technology
 University Heights, Newark, NJ 07102, USA

 Permission to use, copy, modify, and distribute this
 software and its documentation for any purpose and without
 fee is hereby granted, provided that this copyright
 notice appears in all copies. Programmer(s) makes no
 representations about the suitability of this
 software for any purpose.  It is provided "as is" without
 express or implied warranty.

 This script runs all training processes for flare class: C, M, M5 and time window: 24, 48, 72
 using the transformer-based SolarKnowledge model.
 @author: Yasser Abduallah
'''

from SolarKnowledge_model import SolarKnowledge
from utils import get_training_data, data_transform, log, supported_flare_class
import os
import warnings
warnings.filterwarnings('ignore')


def train(time_window, flare_class):
    log('Training is initiated for time window: ' + str(time_window) + ' and flare class: ' + flare_class, verbose=True)

    # Load training data and transform the labels (one-hot encoding)
    X_train, y_train = get_training_data(time_window, flare_class)
    y_train_tr = data_transform(y_train)

    epochs = 20
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Create an instance of the SolarKnowledge transformer-based model
    model = SolarKnowledge(early_stopping_patience=3)
    model.build_base_model(input_shape)  # You can pass additional parameters if needed
    model.compile()

    # Train the model (the new model outputs (batch, num_classes) so no additional reshape is needed)
    model.fit(X_train, y_train_tr, epochs=epochs, verbose=2)

    # Construct a directory path for saving the weights
    w_dir = os.path.join('models', str(time_window), str(flare_class))
    model.save_weights(flare_class=flare_class, w_dir=w_dir)


if __name__ == '__main__':
    # Loop over the defined time windows and flare classes
    for time_window in [24, 48, 72]:
        for flare_class in ['C', 'M', 'M5']:
            if flare_class not in supported_flare_class:
                print('Unsupported flare class:', flare_class, 'It must be one of:', ', '.join(supported_flare_class))
                continue
            train(str(time_window), flare_class)
            log('===========================================================\n\n', verbose=True)
