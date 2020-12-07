from mpi4py import MPI
import numpy as np
import json
import os
import subprocess
import sys
from env import make_env
from controller import make_controller, simulate
from es import CMAES, SimpleGA, OpenES, PEPG
from utils import PARSER
import argparse
import time
import random
import gc
from rnn.rnn import MDNRNN
args = PARSER.parse_args()
optimizer = args.controller_optimizer
num_episode = args.controller_num_episode
num_test_episode = args.controller_num_test_episode
eval_steps = args.controller_eval_steps
num_worker = args.controller_num_worker
num_worker_trial = args.controller_num_worker_trial
antithetic = (args.controller_antithetic == 1)
if antithetic and optimizer != 'oes':
  raise ValueError('OpenES is the only optimizer we support antithetic sampling')
retrain_mode = (args.controller_retrain == 1)
cap_time_mode= (args.controller_cap_time == 1)
seed_start = args.controller_seed_start
env_name = args.env_name
exp_name = args.exp_name
batch_mode = args.controller_batch_mode
controller = make_controller(args=args)
controller.load_model("results/WorldModels/CarRacing-v0/log/CarRacing-v0.cma.16.8.best.json")
env = make_env(args=args, dream_env=args.dream_env)
reward_list, t_list = simulate(controller, env)
print(reward_list)
