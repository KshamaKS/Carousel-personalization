# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 13:29:41 2021

@author: Admin
"""
from collections import defaultdict
from math import log
from scipy.special import expit
from scipy.optimize import minimize
import numpy as np

class RandomPolicy():
    def __init__(self, n_playlists, cascade_model=True):
        self.cascade_model = cascade_model
        self.n_playlists = n_playlists

    def recommend_to_users_batch(self, batch_users, n_recos=12, l_init=3):
        n_users = len(batch_users)
        recos = np.zeros((n_users, n_recos), dtype=np.int64)
        r = np.arange(self.n_playlists)
        for i in range(n_users):
            np.random.shuffle(r)
            recos[i] = r[:n_recos]
        return recos

    def update_policy(self, user_ids, recos, rewards, l_init=3):
        return
