import gym
from projects.imitation.expert_agent import ExpertAgent
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

env = gym.make('Ant-v2')

sess = tf.Session()
agent = ExpertAgent('Ant.pkl')

# init vars
# init_vars = [tf.global_variables_initializer(), tf.local_variables_initializer()]
sess.run(tf.global_variables_initializer())

total_reward = 0
observation = env.reset()

for _ in range(1000):
    env.render()
    # random_action = env.action_space.sample()
    expert_action = agent.actor.get_action(observation, sess)
    observation, reward, done, info = env.step(expert_action)
    total_reward += reward
    if done:
        break

print('Total reward = {}'.format(total_reward))
env.close()
sess.close()
