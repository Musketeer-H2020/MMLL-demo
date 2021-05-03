# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disables tensorflow warnings
import tensorflow as tf
import sys
sys.path.append("../../../../")

from demo_tools.data_connectors.Load_from_file import Load_From_File as DC

# Load data
dataset = 'mnist'
dc = DC('../../../../input_data/' + dataset + '_demonstrator_data.pkl')
[Xtst, ytst] = dc.get_data_tst()

# Load model
filename = 'results/models/Master_' + dataset + '_model'
model = tf.keras.models.load_model(filename)

# Make predictions on test data
preds_tst = model.predict(Xtst)
print(preds_tst)
