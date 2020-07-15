import gym
from projects.imitation.expert_agent import ExpertAgent


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


def main():
    env = gym.make('Ant-v2')
    play_tf(env)
    env.close()


if __name__ == "__main__":
    main()
