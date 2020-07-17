import pickle
import gym
from projects.imitation.bc_agent import BCAgent
from projects.imitation.expert_agent import ExpertAgent
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def play(env, agent, num_episodes=1000):
    total_reward = 0
    observation = env.reset()
    for _ in range(num_episodes):
        env.render()
        # random_action = env.action_space.sample()
        action = agent.actor.get_action(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
    print(total_reward)


def play_tf(env, agent, num_episodes=1000):
    total_reward = 0
    observation = env.reset()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for _ in range(num_episodes):
        env.render()
        action = agent.actor.get_action(observation, sess)
        observation, reward, done, info = env.step(action)
        total_reward += reward
    print(total_reward)
    sess.close()


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


def dagger(env, agent, expert, dagger_steps=5, num_iters=1000):
    observation = env.reset()
    for _ in range(dagger_steps):
        paths = []
        # collect paths with current policy (pi_theta)
        for _ in range(num_iters):
            action = agent.actor.get_action(observation)
            next_observation, reward, done, info = env.step(action)
            paths.append((observation, action, reward, next_observation, done))

        # relabel collected paths with expert policy
        # ...

        # train agent
        train_agent(agent)


def main():
    do_dagger = True
    env = gym.make('Ant-v2')
    agent = BCAgent()
    # initial training with expert data
    load_expert_data_to_buffer(agent)
    train_agent(agent, num_iters=1000, batch_size=1000)
    expert_agent = ExpertAgent('Ant.pkl')
    if do_dagger:
        dagger(env, agent, expert_agent)
    play(env, agent)
    env.close()


if __name__ == "__main__":
    main()
