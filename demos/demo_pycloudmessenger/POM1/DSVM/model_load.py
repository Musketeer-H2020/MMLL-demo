# -*- coding: utf-8 -*-

import pickle
import sys
sys.path.append("../../../../")

from demo_tools.data_connectors.Load_from_file import Load_From_File as DC

# Load data
dataset = 'mnist_binclass'
dc = DC('../../../../input_data/' + dataset + '_demonstrator_data.pkl')
[Xtst, ytst] = dc.get_data_tst()
Xtst = Xtst / 255.0

# Load model
with open('results/models/Master_Marcos_' + dataset + '_model.pkl', 'rb') as f:
    model = pickle.load(f)


# Make predictions on test data
preds_tst = model.predict(Xtst)
print(preds_tst)
