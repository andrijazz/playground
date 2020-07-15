import pickle
import gym
from projects.imitation.expert_agent import ExpertAgent
from projects.imitation.mlp_agent import MLPAgent


def play_tf(env, num_episodes=1000):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    agent = ExpertAgent('Ant.pkl')
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


def train_agent(agent, num_iters, batch_size):
    # populate replay buffer
    with open('expert_data_Ant-v2.pkl', 'rb') as data:
        loaded_paths = pickle.load(data)
        # add collected data to replay buffer
        agent.add_to_replay_buffer(loaded_paths)

    # train agent (using sampled data from replay buffer)
    for _ in range(num_iters):
        ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = agent.sample(batch_size)
        agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)


def main():
    env = gym.make('Ant-v2')
    # agent = MLPAgent()
    # train_agent(agent, num_iters=1000, batch_size=1000)
    # play(env, agent)
    # play_tf(env)
    env.close()


if __name__ == "__main__":
    main()
