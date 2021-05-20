# -*- coding: utf-8 -*-
'''
@author:  Angel Navia Vázquez
Febr. 2021
python3 cr_test.py

'''
# Add higher directory to python modules path.
import sys
import numpy as np

sys.path.append("../../../../")
from demo_tools.crypto.crypt_PHE import Crypto as CR

# Defining encryption object
key_size = 512
cr = CR(key_size=key_size)

NP = 200
NI = 20

X = np.random.normal(0, 1, (NP, NI))
y = np.random.normal(0, 1, (NP, 1))
w = np.random.normal(0, 1, (NI, 1))

o_orig = np.dot(X, w)
e_orig = y - o_orig
Xe_orig = X * e_orig
grad_e_orig = np.sum(Xe_orig, axis=0).reshape((-1, 1))

#grad_orig = np.dot(e_orig.T, X).reshape(-1, 1)

w_encr = cr.encrypter.encrypt(w)

err = np.linalg.norm(w - cr.decrypter.decrypt(w_encr))
print('Err in w ', err)

o_encr = np.dot(X, w_encr)
err = np.linalg.norm(o_orig - cr.decrypter.decrypt(o_encr))
print('Err in o ', err)

e_encr = (y - o_encr)
err = np.linalg.norm(e_orig - cr.decrypter.decrypt(e_encr))
print('Err in e ', err)

Xe_encr = X * e_encr
err = np.linalg.norm(Xe_orig - cr.decrypter.decrypt(Xe_encr))
print('Err in Xe ', err)

grad_e_encr = np.sum(Xe_encr, axis=0).reshape((-1, 1))
err = np.linalg.norm(grad_e_orig - cr.decrypter.decrypt(grad_e_encr))
print('Err in grad ', err)

