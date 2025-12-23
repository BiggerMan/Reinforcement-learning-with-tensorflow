"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            # 直接使用 pd.concat 替代 append 添加新状态
            new_state = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=state,
            )
            # 将新状态转换为DataFrame并与原q_table合并
            self.q_table = pd.concat(
                [self.q_table, new_state.to_frame().T],
                ignore_index=False
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass


# backward eligibility traces
class SarsaLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        # backward view, eligibility trace.
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            to_be_append = pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            self.q_table = pd.concat(
                [self.q_table, to_be_append.to_frame().T],
                ignore_index=False
            )

            # also update eligibility trace
            self.eligibility_trace = pd.concat(
                [self.eligibility_trace, to_be_append.to_frame().T],
                ignore_index=False
            )

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal' and s_ != 'hole':
            # self.q_table.loc[s_, a_]：s_ 行 a_ 列的 Q 值
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        error = q_target - q_predict

        # increase trace amount for visited state-action pair

        # Method 1:
        # self.eligibility_trace.loc[s, a] += 1

        # Method 2:
        # 强化学习中的资格迹（Eligibility Trace）机制，用于Sarsa (λ) 算法中，
        # 核心作用是追踪近期影响过 Q 值的 状态+动作 对（S,A），并根据反馈误差对这些状态 - 动作对的 Q 值进行高效更新。
        self.eligibility_trace.loc[s, :] *= 0
        self.eligibility_trace.loc[s, a] = 1

        # Q update
        self.q_table += self.lr * error * self.eligibility_trace

        # decay eligibility trace after update
        self.eligibility_trace *= self.gamma*self.lambda_

    def print_q_table(self):
        print(self.q_table)

    def print_eligibility_trace(self):
        print(self.eligibility_trace)

    def save_q_table(self, file_path='q_table.csv'):
        self.q_table.to_csv(file_path)

    def save_eligibility_trace(self, file_path='eligibility_trace.csv'):
        self.eligibility_trace.to_csv(file_path)
