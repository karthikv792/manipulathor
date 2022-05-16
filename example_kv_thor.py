from kv_thor import *
import argparse
from addict import Dict
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("env",type=str,default='kv_thor',help="select from [\'kv_thor\']")

    sys_args = Dict()
    args, unknown = parser.parse_known_args()
    for k, v in vars(args).items():
        sys_args[k] = v
    return sys_args

def main():
    args = get_args()
    if args.env == 'kv_thor':
        env = KVThorEnv()
    else:
        raise NotImplementedError
    env.reset()
    total_reward = 0
    step = 0
    while True:
        print("---------------- Step: {} ----------------".format(step))
        if step<10:
            action = 11
        else:
            action = env.action_space().sample()

        obs, reward, done, info = env.step(action)
        print("[MAIN INFO] Observation: {}".format(obs['pickedup_object']))
        print("[MAIN INFO] Action: {}".format(action))
        print("[MAIN INFO] Reward: {}".format(reward))
        print("[MAIN INFO] Done: {}".format(done))
        print("[MAIN INFO] Info: {}".format(info))
        print("[MAIN INFO] Arm dist from object: {}".format(env.arm_distance_from_obj()))
        print("[MAIN INFO] Obj dist from goal: {}".format(env.obj_distance_from_goal()))

        env.render()
        total_reward += reward
        if done:
            break
        step += 1
    print(f"total_reward: {total_reward}")


if __name__ == "__main__":
    main()


