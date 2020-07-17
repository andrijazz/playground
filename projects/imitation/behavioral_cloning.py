import argparse
import imageio
import pickle
import gym
import numpy as np
import os
from projects.imitation.bc_agent import BCAgent
from projects.imitation.expert_agent import ExpertAgent

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def play(env, agent, num_episodes=100):
    reward_per_episode = []
    for _ in range(num_episodes):
        total_reward = 0
        observation = env.reset()
        while True:
            # random_action = env.action_space.sample()
            action = agent.actor.get_action(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        reward_per_episode.append(total_reward)

    return np.mean(reward_per_episode)


def play_tf(env, agent, sess, num_episodes=100):
    reward_per_episode = []
    for _ in range(num_episodes):
        total_reward = 0
        observation = env.reset()
        while True:
            action = agent.actor.get_action(observation, sess)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        reward_per_episode.append(total_reward)
    return np.mean(reward_per_episode)


def load_expert_data_to_buffer(agent):
    # populate replay buffer
    with open('expert_data_Ant-v2.pkl', 'rb') as data:
        loaded_paths = pickle.load(data)
        # add collected data to replay buffer
        agent.add_to_replay_buffer(loaded_paths)


def train_agent(agent, num_iters, batch_size):
    # train agent (using sampled data from replay buffer)
    for _ in range(num_iters):
        ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = agent.sample(batch_size)
        agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)


def dagger(env, agent, expert, sess, max_path_length=500):
    observation = env.reset()
    path = dict(
        observation=list(),
        action=list(),
        reward=list(),
        next_observation=list(),
        terminal=list()
    )

    # collect paths with current policy (pi_theta)
    step = 0
    for _ in range(max_path_length):
        action = agent.actor.get_action(observation)
        next_observation, reward, done, info = env.step(action)
        path['observation'].append(observation)
        path['reward'].append(reward)
        path['next_observation'].append(next_observation)
        path['terminal'].append(done)
        step += 1
        observation = next_observation

        if done:
            break

    # relabel collected paths with expert policy
    for i in range(step):
        action = expert.actor.get_action(path['observation'][i], sess)
        path['action'].append(action.reshape(-1))

    agent.add_to_replay_buffer([path])

    # retrain agent with new data
    train_agent(agent, num_iters=1000, batch_size=1000)


def main(args):
    env = gym.make('Ant-v2')
    agent = BCAgent()
    # initial training on BCAgent with expert data
    load_expert_data_to_buffer(agent)
    train_agent(agent, num_iters=1000, batch_size=1000)
    print('Initial BCAgent avg reward = {}'.format(play(env, agent)))

    # Expert agent results
    expert_agent = ExpertAgent('Ant.pkl')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print('Expert agent avg reward = {}'.format(play_tf(env, expert_agent, sess)))

    # DAgger
    if args.do_dagger:
        # repeat dagger 5 times
        for _ in range(5):
            dagger(env, agent, expert_agent, sess)
        # print reward after dagger
        print('BCAgent after dagger avg reward = {}'.format(play(env, agent)))

    if args.log_video:
        step = 0
        max_frames = 128
        observation = env.reset()
        frames = []
        while True:
            frame = env.sim.render(camera_name='track', height=500, width=500)[::-1]
            action = agent.actor.get_action(observation)
            observation, reward, done, info = env.step(action)
            frames.append(frame)
            step += 1
            if done or step > max_frames:
                break
        current_dir = os.path.dirname(os.path.abspath(__file__))
        out_file = os.path.join(current_dir, 'movie.gif')
        imageio.mimsave(out_file, frames)
    env.close()
    sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_dagger', type=bool, help='Run dagger', default=True)
    parser.add_argument('--log_video', type=bool, help='Create video file of agent', default=True)
    args = parser.parse_args()
    main(args)
