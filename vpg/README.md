## Vanilla Policy Gradient algorithm

Single file pytorch implementation of Reinforce (Vanilla Policy Gradient) algorithm along with reward-to-go and baseline improvements.

### Math
$\pi_{\theta}(a|s)$ is the policy function that outputs the probability of taking action $a$ in state $s$ (neural network taking the state as an input and outputs the probability of each action).

$p(s'|s, a)$ is the transition probability of going from state $s$ to state $s'$ by taking action $a$.

$\tau = (s_1, a_1 ... s_T, a_T)$ is a trajectory.

$p_{\theta}(s_1, a_1, ..., s_{T}, a_{T})$ is the trajectory probability distribution over sequence of states and actions.

- [x] Working version
- [ ] Math doc 
- [x] Reward-to-go
- [ ] Baselines
- [ ] Result graphs
- [ ] Parallelize trajectory collection (ray) 
- [ ] GAE-lambda

<!--
This [document]() describes the math behind the algorithm.
Math behind the algorithm:
```.env
...
```

Improvement 1. Reward-to-go
```.env
...
```
Improvement 2. Baselines
```.env
...
```
-->

### Results
<!-- WANDB graphs / Videos -->

### References
* [Levine, 2020, CS285 Policy Gradients](https://youtu.be/GKoKNYaBvM0)
* [Sutton, 2000, Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)

<!--
Other references

- lazy frame https://www.linkedin.com/posts/aleksagordic_deeplearning-project-update-activity-6786939777359912960-ETeR
- https://github.com/transedward/pytorch-dqn
-
-->
