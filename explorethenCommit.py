# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 13:24:52 2021

@author: Admin
"""


from collections import defaultdict
from math import log
from scipy.special import expit
from scipy.optimize import minimize
import numpy as np


class ExploreThenCommit():
    def __init__(self, user_segment, n_playlist, min_n):
        self.user_segment = user_segment
        self.min_n = min_n
        self.n_segments = len(np.unique(self.user_segment))
        self.playlist_display = np.zeros((self.n_segments, n_playlist))
        self.playlist_success = np.zeros((self.n_segments, n_playlist))
        
    
    def recommend_to_batch(self, batch_ids, n_recos=12, l_init=3):
        segments = np.take(self.user_segment, batch_ids, axis = 0)
        success = np.take(self.playlist_success, segments, axis = 0)
        display = np.take(self.playlist_display, segments, axis = 0)
        user_random_score = np.random.random(display.shape)
        score = np.divide(success, display, out = np.zeros_like(display), where = display!=0)   #calculate the average score of each playlist for each segment
        discounted_displays = np.maximum(np.zeros_like(display), self.min_n - display)
        #pick the first 12 playlist which have been picked least number of times based on -discounted_displays
        #incase there is a tie, use -score as tiebreaker, if the tie exists use user_random_score
        choice = np.lexsort((user_random_score, -score, -discounted_displays))[:,:n_recos]
        np.random.shuffle(choice[0:l_init])
        return choice
    
    def update_policy(self, user_ids, recos, rewards):
        batch_users = len(user_ids)
        for i in range(batch_users):
            user_segment = self.user_segment[user_ids[i]]
            for p, r in zip(recos[i], rewards[i]):
                self.playlist_success[user_segment][p] += r
                self.playlist_display[user_segment][p] += 1
                
        return