# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:05:18 2021

@author: Admin
"""

import json
import logging
import numpy as np
import pandas as pd
import time
from scipy.special import expit
from functools import reduce
from explorethenCommit import ExploreThenCommit
from UCB import KLUCBSegmentPolicy
from randomPolicy import RandomPolicy

from ep_greedy import EpsilonGreedySegmentPolicy


logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt

playlists_path = "C:/Users/Admin/Documents/grad_school/sem1/odm/project/playlist_features.csv"
users_path = "C:/Users/Admin/Documents/grad_school/sem1/odm/project/user_features.csv"
logger.info("LOADING DATA")
logger.info("Loading playlist data")
playlists_df = pd.read_csv(playlists_path)
playlist_features = np.array(playlists_df)

logger.info("Loading user data\n")
users_df = pd.read_csv(users_path)
n_users = len(users_df)

n_len = len(playlists_df)
print(playlists_df.shape)
print(users_df.shape)

user_features = np.array(users_df.drop(["segment"], axis = 1))
user_features = np.concatenate([user_features, np.ones((n_users,1))], axis = 1)

user_segment = np.array(users_df.segment)
n_recos = 12

class contextEnv():
    def __init__(self, user_features, playlist_features, user_segment, n_recos):
        self.users_features = user_features
        self.playlist_features = playlist_features
        self.user_segment = user_segment
        self.n_recos = n_recos
        self.x_reward = np.zeros(user_features.shape[0])
        self.x_segment_reward = np.zeros(user_features.shape[0])
        self.compute_optimal_reward_all()
        self.compute_optimal_rewards_segment()
        
        
    def compute_optimal_recos(self, user_ids, n_recos):
        #pick top n_recos playlist for the batch of users: take dot product of the user and playlist features
        batch_users = np.take(self.users_features,  user_ids, axis  = 0)
        prod = batch_users.dot(self.playlist_features.T)
        optimal_playlists = np.argsort(-prod)[:, :n_recos]
        return optimal_playlists
        
    def compute_theoretical_reward(self, batch_user_ids, playlists_selected):
        batch_users = np.take(self.users_features, batch_user_ids, axis =0)
        optimal_recos = np.take(self.playlist_features, playlists_selected, axis = 0)
        n_users = len(batch_user_ids)
        th_reward = np.zeros(n_users) 
        for i in range(n_users):
            probas = expit(batch_users[i].dot(optimal_recos[i].T))
            th_reward[i] = 1 - reduce(lambda x,y: x*y,1-probas)
        return th_reward
        
    
    def compute_optimal_reward_all(self):
        n_users = self.users_features.shape[0]
        u = 0
        batch = 10000
        while(u<n_users):
            user_ids = range(u, min(n_users, u + batch))
            optimal = self.compute_optimal_recos(user_ids, self.n_recos)
            self.x_reward[u:min(n_users, u + batch)] = self.compute_theoretical_reward(user_ids, optimal)
            u += batch
        return
    
    def compute_optimal_segment_recos(self, n):
        n_segments = len(np.unique(self.user_segment))
        recos_segment = np.zeros((n_segments, n),  dtype = np.int64)
        for i in range(n_segments):
            users = np.take(self.users_features, np.where(self.user_segment == i)[0], axis = 0 )
            prod = np.mean(expit(users.dot(self.playlist_features.T)), axis = 0)
            recos_segment[i] = np.argsort(-prod)[:n]
        return recos_segment
            
    
    def compute_optimal_rewards_segment(self):
        n_users= self.users_features.shape[0]
        u =0 
        batch = 10000
        segment_recos = self.compute_optimal_segment_recos(self.n_recos)
        while (u<n_users):
            users_ids = range(u, min(n_users, u+ batch))
            user_segment = np.take(self.user_segment, users_ids, axis = 0)
            opt_recos = np.take(segment_recos, user_segment, axis = 0)
            opt_rewards = self.compute_theoretical_reward(users_ids, opt_recos)
            self.x_segment_reward[u:min(n_users, u+ batch)] = opt_rewards
            u += batch
        return 
    
    def batch_user_rewards(self, batch_user_ids, batch_recos):
        batch_users = np.take(self.users_features, batch_user_ids, axis =0)
        batch_playlist_features = np.take(self.playlist_features, batch_recos, axis = 0)
        n_users = len(batch_users)
        n = len(batch_recos[0])
        probas = np.zeros((n_users, n))
        for i in range(n_users):
            #probability to stream 
            probas[i] = expit(batch_users[i].dot(batch_playlist_features[i].T))
        rewards = np.zeros((n_users, n))
        rewards_uncascaded = np.random.binomial(1, probas) # drawing rewards from probabilities
        positive_rewards = set()
        nz = rewards_uncascaded.nonzero()
        
        for i in range(len(nz[0])):
            if nz[0][i] not in positive_rewards:
                rewards[nz[0][i]][nz[1][i]] = 1
                positive_rewards.add(nz[0][i])
        return rewards
        
  
n_rounds = 100
regret_explore = np.zeros((n_rounds))
n_users_per_round =  20000
overall_optimal_reward_explore_commit = np.zeros((n_rounds))
overall_rewards_explore_commit = np.zeros((n_rounds))


env = contextEnv(user_features, playlist_features, user_segment, n_recos)  
exp = ExploreThenCommit(user_segment, len(playlist_features), 100)

'''
for i in range(n_rounds):
    user_ids = np.random.choice(range(n_users), n_users_per_round)
    overall_optimal_reward_explore_commit[i] = np.take(env.x_reward, user_ids).sum()
    recos = exp.recommend_to_batch(user_ids)
    rewards = env.batch_user_rewards(batch_user_ids= user_ids, batch_recos=recos)
    exp.update_policy(user_ids, recos, rewards)
    overall_rewards_explore_commit[i] = rewards.sum()
    regret_explore[i] = np.sum(overall_optimal_reward_explore_commit - overall_rewards_explore_commit)

    print("Cumulative regrets:", np.sum(overall_optimal_reward_explore_commit - overall_rewards_explore_commit)) 


t = range(100)
plt.show()
'''

eps = [1e-10, 1e-15, 1e-20, 1e-25, 1e-5]

regret_UCB_all = np.zeros((5, n_rounds))
t = 0

for ep in eps:
    print(ep)
    exp = KLUCBSegmentPolicy(user_segment, len(playlist_features), eps = ep)
    overall_optimal_reward = np.zeros((n_rounds))
    overall_rewards_UCB = np.zeros((n_rounds))

    
    for i in range(n_rounds):
        user_ids = np.random.choice(range(n_users), n_users_per_round)
        overall_optimal_reward[i] = np.take(env.x_reward, user_ids).sum()
        recos = exp.recommend_to_users_batch(user_ids)
        rewards = env.batch_user_rewards(batch_user_ids= user_ids, batch_recos=recos)
        exp.update_policy(user_ids, recos, rewards)
        overall_rewards_UCB[i] = rewards.sum()
        regret_UCB_all[t, i] = np.sum(overall_optimal_reward - overall_rewards_UCB)
        
        #print("Cumulative regrets:", np.sum(overall_optimal_reward - overall_rewards_UCB)) 
    t += 1
'''
exp = RandomPolicy(len(playlist_features))
overall_optimal_reward = np.zeros((n_rounds))
overall_rewards_random = np.zeros((n_rounds))
regret_random = np.zeros((n_rounds))

print('random')

for i in range(n_rounds):
    user_ids = np.random.choice(range(n_users), n_users_per_round)
    overall_optimal_reward[i] = np.take(env.x_reward, user_ids).sum()
    recos = exp.recommend_to_users_batch(user_ids)
    rewards = env.batch_user_rewards(batch_user_ids= user_ids, batch_recos=recos)
    exp.update_policy(user_ids, recos, rewards)
    overall_rewards_random[i] = rewards.sum()
    regret_random[i] = np.sum(overall_optimal_reward - overall_rewards_random)

    print("Cumulative regrets:", i, ":", np.sum(overall_optimal_reward - overall_rewards_random)) 

exp = EpsilonGreedySegmentPolicy(user_segment, len(playlist_features), epsilon = 0.1)
overall_optimal_reward = np.zeros((n_rounds))
overall_rewards_epsilon = np.zeros((n_rounds))
regret_epsilon = np.zeros((n_rounds))

print('random')

for i in range(n_rounds):
    user_ids = np.random.choice(range(n_users), n_users_per_round)
    overall_optimal_reward[i] = np.take(env.x_reward, user_ids).sum()
    recos = exp.recommend_to_users_batch(user_ids)
    rewards = env.batch_user_rewards(batch_user_ids= user_ids, batch_recos=recos)
    exp.update_policy(user_ids, recos, rewards)
    overall_rewards_epsilon[i] = rewards.sum()
    regret_epsilon[i] = np.sum(overall_optimal_reward - overall_rewards_epsilon)

    print("Cumulative regrets:", i, ":", np.sum(overall_optimal_reward - overall_rewards_epsilon)) 



t = range(100)
plt.plot(t, regret_UCB, label= 'UCB')
plt.plot(t, regret_explore, label = 'Explore')
plt.plot(t, regret_random, label = 'Random')
plt.plot(t, regret_epsilon, label = 'epsilon')
plt.legend()
plt.title(' evaluation of top-12 playlist recommendation: expected cumulative regrets \n of policies over 100 simulated rounds.')
plt.xlabel('Rounds')
plt.ylabel('cumulative Regret')
plt.show()

'''

t = range(100)
plt.plot(t, regret_UCB_all[0], label= eps[0])
plt.plot(t, regret_UCB_all[1], label = eps[1])
plt.plot(t, regret_UCB_all[2], label = eps[2])
plt.plot(t, regret_UCB_all[3], label = eps[3])
plt.plot(t, regret_UCB_all[4], label = eps[4])
plt.legend()
plt.title(' evaluation of UCB Algorithm with different epsilon values: \n expected cumulative regrets of UCB over 100 simulated rounds.')
plt.xlabel('Rounds')
plt.ylabel('cumulative Regret')
plt.show()