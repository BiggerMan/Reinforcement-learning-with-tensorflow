"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import os
import ast


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, q_table_file='q_table.csv'):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table_file = q_table_file

        # 尝试加载已有的Q表，如果不存在则创建新的
        self.q_table = self.load_q_table()

    def load_q_table(self):
        """从文件加载Q表，如果文件不存在则创建新的空表"""
        if os.path.exists(self.q_table_file):
            print(f"Loading Q-table from {self.q_table_file}")
            try:
                # 读取CSV文件
                q_table = pd.read_csv(self.q_table_file, index_col=0)

                # 修复索引格式问题
                new_index = []
                for idx in q_table.index:
                    if idx == "terminal":
                        new_index.append("terminal")
                    else:
                        try:
                            # 解析字符串格式的状态（如"[5.0, 245.0, 35.0, 275.0]"）
                            parsed_idx = ast.literal_eval(idx)
                            # 如果是列表，转换为元组（因为列表不可哈希，不能作为索引）
                            if isinstance(parsed_idx, list):
                                new_index.append(tuple(parsed_idx))
                            else:
                                new_index.append(parsed_idx)
                        except:
                            new_index.append(idx)

                q_table.index = new_index

                # 修复列名格式问题 - 将列名转换为整数
                new_columns = []
                for col in q_table.columns:
                    try:
                        new_columns.append(int(col))
                    except:
                        new_columns.append(col)
                q_table.columns = new_columns

                # 确保列名与actions一致
                if set(self.actions) == set(q_table.columns):
                    print("Q-table loaded successfully!")
                    return q_table
                else:
                    print(f"Actions mismatch: expected {set(self.actions)}, got {set(q_table.columns)}")
                    return pd.DataFrame(columns=self.actions, dtype=np.float64)
            except Exception as e:
                print(f"Error loading Q-table: {e}, creating new one")
                return pd.DataFrame(columns=self.actions, dtype=np.float64)
        else:
            print(f"No existing Q-table found at {self.q_table_file}, creating new one")
            return pd.DataFrame(columns=self.actions, dtype=np.float64)

    def save_q_table(self):
        """保存Q表到文件"""
        try:
            # 在保存前，确保索引都是字符串格式
            q_table_to_save = self.q_table.copy()

            # 转换索引为字符串
            new_index = []
            for idx in q_table_to_save.index:
                if idx == "terminal":
                    new_index.append("terminal")
                else:
                    # 将元组或列表转换为字符串，但不带额外的引号
                    new_index.append(str(list(idx)) if isinstance(idx, tuple) else str(idx))

            q_table_to_save.index = new_index

            # 确保列名是字符串，以便正确保存
            q_table_to_save.columns = q_table_to_save.columns.astype(str)

            q_table_to_save.to_csv(self.q_table_file)
            print(f"Q-table saved to {self.q_table_file}")
        except Exception as e:
            print(f"Error saving Q-table: {e}")

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal' and s_ != 'hole':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
        if s_ == 'terminal':
            print(f"Q-table after update:\n{self.q_table.to_string()}")


    def check_state_exist(self, state):
        if state not in self.q_table.index:
            print(f"State {state} not found in Q-table, adding it")
            # 使用 pd.concat 替代 append 添加新状态
            new_state = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=state
            )
            # 将新状态转换为DataFrame并与原q_table合并
            self.q_table = pd.concat(
                [self.q_table, new_state.to_frame().T],
                ignore_index=False
            )

    def display_q_table(self):
        print(self.q_table)
