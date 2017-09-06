import build_graph
import numpy as np
import tensorflow as tf


class Agent:
    def __init__(self, model, icm_model, num_actions, name='global', lr=2.5e-4, gamma=0.99):
        self.num_actions = num_actions
        self.gamma = gamma
        self.t = 0
        self.name = name

        act, train, update_local, state_value, bonus = build_graph.build_train(
            model=model,
            icm_model=icm_model,
            num_actions=num_actions,
            optimizer=tf.train.AdamOptimizer(learning_rate=1e-4),
            scope=name
        )

        self._act = act
        self._train = train
        self._update_local = update_local
        self._state_value = state_value
        self._bonus = bonus

        self.initial_state = np.zeros((1, 256), np.float32)
        self.rnn_state0 = self.initial_state
        self.rnn_state1 = self.initial_state
        self.last_obs = None
        self.last_reward = None
        self.last_action = None
        self.last_value = None

        self.states = []
        self.next_states = []
        self.rewards = []
        self.actions = []
        self.values = []

    def train(self, bootstrap_value, summary_writer):
        states = np.array(self.states, dtype=np.float32) / 255.0
        next_states = np.array(self.next_states, dtype=np.float32) / 255.0
        actions = np.array(self.actions, dtype=np.uint8)
        returns = []
        R = bootstrap_value
        for r in reversed(self.rewards):
            R = r + 0.99 * R
            returns.append(R)
        returns = np.array(list(reversed(returns)), dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)

        advantages = returns - values

        summary, loss = self._train(states, next_states, self.initial_state,
                self.initial_state, actions, returns, advantages)
        summary_writer.add_summary(summary, loss)
        self._update_local()
        return loss

    def act(self, obs):
        normalized_obs = np.zeros((1, 42, 42, 4), dtype=np.float32)
        normalized_obs[0] = np.array(obs, dtype=np.float32) / 255.0
        prob, rnn_state = self._act(normalized_obs, self.rnn_state0, self.rnn_state1)
        action = np.random.choice(range(self.num_actions), p=prob[0])
        self.rnn_state0, self.rnn_state1 = rnn_state
        return action

    def act_and_train(self, obs, reward, summary_writer):
        prob, rnn_state = self._act(self.process_obs(obs), self.rnn_state0, self.rnn_state1)
        action = np.random.choice(range(self.num_actions), p=prob[0])
        value = self._state_value(self.process_obs(obs), self.rnn_state0, self.rnn_state1)[0][0]

        if len(self.states) == 20:
            self.train(self.last_value, summary_writer)
            self.states = []
            self.next_states = []
            self.rewards = []
            self.actions = []
            self.values = []

        if self.last_obs is not None:
            reward += 288.0 * self._bonus(self.process_obs(self.last_obs),
                    self.process_obs(obs), [self.last_action])
            self.states.append(self.last_obs)
            self.next_states.append(obs)
            self.rewards.append(reward)
            self.actions.append(self.last_action)
            self.values.append(self.last_value)

        self.t += 1
        self.rnn_state0, self.rnn_state1 = rnn_state
        self.last_obs = obs
        self.last_reward = reward
        self.last_action = action
        self.last_value = value
        return action

    def stop_episode_and_train(self, obs, reward, summary_writer, done=False):
        self.states.append(self.last_obs)
        self.next_states.append(obs)
        self.rewards.append(reward)
        self.actions.append(self.last_action)
        self.values.append(self.last_value)
        self.train(0, summary_writer)
        self.stop_episode()

    def stop_episode(self):
        self.rnn_state0 = self.initial_state
        self.rnn_state1 = self.initial_state
        self.last_obs = None
        self.last_reward = None
        self.last_action = None
        self.last_value = None
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.values = []

    def process_obs(self, obs):
        normalized_obs = np.zeros((1, 42, 42, 4), dtype=np.float32)
        normalized_obs[0] = np.array(obs, dtype=np.float32) / 255.0
        return normalized_obs
