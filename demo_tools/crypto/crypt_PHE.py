# -*- coding: utf-8 -*-
'''
Encryption under CBP

'''
import numpy as np
import random
import phe  # pip install phe


class Crypto:

    def __init__(self, key_size: int = 3072):

        """
		Create a :class:`Crypto` instance. Generates the encryption parameters.

		Parameters
		----------
		key_size:    length of the key in bits. Default is 3072 to provide strength=128
					 (years 2016 - 2030), according to https://www.keylength.com/en/4/

		"""
        self.keysize = key_size
        self.PK, self.SK = self.generate_keys()

        self.encrypter = Encrypter(self.PK)
        self.decrypter = Decrypter(self.SK)

    def generate_keys(self):
        """
        Generate public and secret key for partial homomorphic encryption.

        :return: A tuple holding the public and secret key
        :rtype: tuple(class:`PaillierPublicKey`, class:`PaillierSecretKey`)
        """
        return phe.paillier.generate_paillier_keypair(n_length=self.keysize)

    def get_encrypter(self):

        return self.encrypter

    def get_decrypter(self):

        return self.decrypter


class Encrypter:

    def __init__(self, PK):
        self.PK = PK
 
    def vE(self, x):
        """
        Encrypt given vector.

        :param x: Vector to be encrypted
        :type x: np.ndarray
        :return: Encrypted vector
        :rtype: array(class:`EncryptedNumber`)
        """
        return np.vectorize(self.basic_encrypt)(x)

    def basic_encrypt(self, x):
        """
        Encrypt a given number x with the key Pk.

        Parameters
        ----------
        xq: integer value to be encrypted

        Returns
        -------
        x_encr: Encrypted value

        """
        return self.PK.encrypt(x)

    def encrypt(self, x):

        if type(x) in [float, int]:
            return self.basic_encrypt(x)
        else:
            return self.vE(x)


class Decrypter:

    def __init__(self, SK):
        self.SK = SK

    def basic_decrypt(self, x_encr):
        """
        Decrypt a given number x_encr with the secret key SK. It uses its own self.SK by default

        Parameters
        ----------
        xq_encr: encrypted number

        Returns
        -------
        xq: Decrypted value

        """
        return self.SK.decrypt(x_encr)

    def vD(self, x):
        """
        Encrypt given vector.

        :param x: Vector to be encrypted
        :type x: np.ndarray
        :return: Encrypted vector
        :rtype: array(class:`EncryptedNumber`)
        """
        return np.vectorize(self.basic_decrypt)(x)

    def decrypt(self, x_encr):

        is_value = True
        try:
            [M, N] = x_encr.shape
            is_value = False
        except:
            pass

        try:
            [M] = x_encr.shape
            is_value = False
        except:
            pass

        if is_value:
            return self.basic_decrypt(x_encr)
        else:
            return self.vD(x_encr)
