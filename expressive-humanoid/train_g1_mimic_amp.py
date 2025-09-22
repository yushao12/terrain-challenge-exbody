#!/usr/bin/env python3
"""
训练G1 Mimic AMP的脚本
"""

import os
import numpy as np
from datetime import datetime
import torch

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

def train_g1_mimic_amp():
    """训练G1 Mimic AMP"""
    
    # 设置参数
    class Args:
        def __init__(self):
            self.task = "g1_mimic_amp"
            self.headless = True
            self.seed = 1
            self.max_iterations = 1500
            self.exptid = f"g1_mimic_amp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.proj_name = "g1_mimic_amp"
            self.debug = False
    
    args = Args()
    
    # 获取任务
    task, env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 设置日志路径
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    os.makedirs(log_pth, exist_ok=True)
    
    print(f"Training G1 Mimic AMP...")
    print(f"Log path: {log_pth}")
    print(f"Number of environments: {env_cfg.env.num_envs}")
    print(f"Motion file: {env_cfg.motion.motion_name}")
    
    # 创建环境
    env = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # 创建训练器
    ppo_runner = train_cfg.runner.runner_class(env, train_cfg, log_dir=log_pth, device=env.device)
    
    # 开始训练
    ppo_runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    
    print("Training completed!")

if __name__ == "__main__":
    train_g1_mimic_amp()