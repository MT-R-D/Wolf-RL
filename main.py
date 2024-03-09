import torch
import random

from RLNetwork import RLNetwork
from Board import Board, WOLF, SHEEP, ONGOING_GAME, WOLF_WON, SHEEP_WON

# This class will representn the training environment
class Environment:
    def __init__(self) -> None:
        # wolf RL parameters
        self.wolf_learning_rate = 0.001
        self.wolf_discount_factor = 0.9
        self.wolf_epsilon = 0.99
        self.wolf_epsilon_decay = 0.9999
        self.wolf_epsilon_min = 0.1
        self.wolf_network = RLNetwork([64, 32,32,32,32,32,32, 4])
        self.wolf_optimizer = torch.optim.Adam(self.wolf_network.parameters(), lr=self.wolf_learning_rate)
        self.wolf_memory = []
        self.wolf_batch_size = 32
        self.wolf_memory_size = 1000
        self.wolf_memory_counter = 0

        self.sheep_learning_rate = 0.001
        self.sheep_discount_factor = 0.9
        self.sheep_epsilon = 0.1
        self.sheep_epsilon_decay = 0.9999
        self.sheep_epsilon_min = 0.1
        self.sheep_network = RLNetwork([64, 32,32,32,32,32,32, 8])
        self.sheep_optimizer = torch.optim.Adam(self.sheep_network.parameters(), lr=self.sheep_learning_rate)
        self.sheep_memory = []
        self.sheep_batch_size = 32
        self.sheep_memory_size = 1000
        self.sheep_memory_counter = 0

        self.wolf_won = 0
        self.sheep_won = 0

        # Board has methods to modify/examine game state
        self.board = Board()
        self.board.reset()

    def wolf_choose_action(self, state: torch.Tensor) -> int:
        if random.random() > self.wolf_epsilon:
            with torch.no_grad():
                action = self.wolf_network(state).argmax().item()
        else:
            action = random.randint(0, 3)
        return action
    
    def sheep_choose_action(self, state: torch.Tensor) -> int:
        if random.random() > self.sheep_epsilon:
            with torch.no_grad():
                action = self.sheep_network(state).argmax().item()
        else:
            action = random.randint(0, 7)
        return action

    def wolf_store_transition(self, state: torch.Tensor, action: int, reward: int, new_state: torch.Tensor) -> None:
        if self.wolf_memory_counter < self.wolf_memory_size:
            self.wolf_memory.append((state, action, reward, new_state))
        else:
            self.wolf_memory[self.wolf_memory_counter % self.wolf_memory_size] = (state, action, reward, new_state)
        self.wolf_memory_counter += 1

    def sheep_store_transition(self, state: torch.Tensor, action: int, reward: int, new_state: torch.Tensor) -> None:
        if self.sheep_memory_counter < self.sheep_memory_size:
            self.sheep_memory.append((state, action, reward, new_state))
        else:
            self.sheep_memory[self.sheep_memory_counter % self.sheep_memory_size] = (state, action, reward, new_state)
        self.sheep_memory_counter += 1

    def wolf_learn(self) -> None:
        if self.wolf_memory_counter < self.wolf_batch_size:
            return
        sample = random.sample(self.wolf_memory, self.wolf_batch_size)
        states, actions, rewards, new_states = zip(*sample)

        states = torch.stack(states)
        new_states = torch.stack(new_states)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        with torch.no_grad():
            target = rewards + self.wolf_discount_factor * self.wolf_network(new_states).max(1)[0]

        prediction = self.wolf_network(states).gather(1, torch.tensor(actions).view(-1, 1)).squeeze()

        loss = torch.nn.functional.mse_loss(prediction, target)
        self.wolf_optimizer.zero_grad()
        loss.backward()
        self.wolf_optimizer.step()

    def sheep_learn(self) -> None:
        if self.sheep_memory_counter < self.sheep_batch_size:
            return
        sample = random.sample(self.sheep_memory, self.sheep_batch_size)
        states, actions, rewards, new_states = zip(*sample)

        states = torch.stack(states)
        new_states = torch.stack(new_states)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        with torch.no_grad():
            target = rewards + self.sheep_discount_factor * self.sheep_network(new_states).max(1)[0]

        prediction = self.sheep_network(states).gather(1, torch.tensor(actions).view(-1, 1)).squeeze()

        loss = torch.nn.functional.mse_loss(prediction, target)
        self.sheep_optimizer.zero_grad()
        loss.backward()
        self.sheep_optimizer.step()

    def calc_sheep_reward(self, reward: int) -> int:
        if reward == WOLF_WON:
            return -10
        elif reward == SHEEP_WON:
            return 10
        else:
            return 0
        
    def calc_wolf_reward(self, reward: int) -> int:
        if reward == WOLF_WON:
            return 10
        elif reward == SHEEP_WON:
            return -10
        else:
            return 0
        
    def wolf_turn_rl(self, state):
        wolf_action = self.wolf_choose_action(state)
        self.board.move_wolf(wolf_action)
        new_state = self.board.to_state()
        reward = self.board.get_reward()
        self.wolf_store_transition(state, wolf_action, self.calc_wolf_reward(reward), new_state)
        self.wolf_learn()
        return new_state, reward

    def sheep_turn_rl(self, state):
        sheep_action = self.sheep_choose_action(state)
        self.board.move_sheep(sheep_action)
        new_state = self.board.to_state()
        reward = self.board.get_reward()
        self.sheep_store_transition(state, sheep_action, self.calc_sheep_reward(reward), new_state)
        self.sheep_learn()
        return new_state, reward

    def sheep_turn_random(self):
        sheep_action = random.randint(0, 7)
        self.board.move_sheep(sheep_action)
        reward = self.board.get_reward()
        return self.board.to_state(), reward

    def wolf_turn_random(self):
        wolf_action = random.randint(0, 3)
        self.board.move_wolf(wolf_action)
        reward = self.board.get_reward()
        return self.board.to_state(), reward

    def train(self, episodes: int) -> None:
        for episode in range(episodes):
            self.board.reset()
            state = self.board.to_state()
            done = False
            while not done:
                # Wolf's turn
                # state, reward = self.wolf_turn_rl(state)
                state, reward = self.wolf_turn_random()

                # Sheep's turn
                state, reward = self.sheep_turn_rl(state)

                #print board
                # self.board.print()
                # input()

                if reward != ONGOING_GAME:
                    self.board.reset()
                    state = self.board.to_state()
                    if reward == WOLF_WON:
                        self.wolf_won += 1
                    elif reward == SHEEP_WON:
                        self.sheep_won += 1
                    done = True

            self.wolf_epsilon = max(self.wolf_epsilon * self.wolf_epsilon_decay, self.wolf_epsilon_min)
            self.sheep_epsilon = max(self.sheep_epsilon * self.sheep_epsilon_decay, self.sheep_epsilon_min)

            if episode % 100 == 0:
                print(f'Episode {episode}')
                print(f'Wolf Epsilon: {self.wolf_epsilon}')
                print(f'Sheep Epsilon: {self.sheep_epsilon}')
                print(f'Wolf wins: {self.wolf_won}')
                print(f'Sheep wins: {self.sheep_won}')

    def save_models(self, wolf_path: str='wolf_model.pt', sheep_path: str='sheep_path.pt') -> None:
        torch.save(self.wolf_network, wolf_path)
        torch.save(self.sheep_network, sheep_path)

    def load_models(self, wolf_path:str = 'wolf_model.pt', sheep_path:str='sheep_model.pt') -> None:
        self.wolf_network = torch.load(wolf_path)
        self.sheep_network = torch.load(sheep_path)

    def play(self) -> None:
        self.board.reset()
        state = self.board.to_state()
        done = False
        while not done:
            # Wolf's turn
            wolf_action = self.wolf_choose_action(state)
            self.board.move_wolf(wolf_action)
            new_state = self.board.to_state()
            reward = self.board.get_reward()
            state = new_state

            # Sheep's turn
            sheep_action = self.sheep_choose_action(state)
            self.board.move_sheep(sheep_action)
            new_state = self.board.to_state()
            reward = self.board.get_reward()
            state = new_state

            #print board
            self.board.print()
            input()

            if reward != ONGOING_GAME:
                self.board.reset()
                state = self.board.to_state()
                done = True
        if reward == WOLF_WON:
            print('Wolf wins!')
        elif reward == SHEEP_WON:
            print('Sheep wins!')

    def free_train(self, save_interval: int, episodes: int, wolf_path: str, sheep_path: str) -> None:
        for episode in range(episodes):
            self.board.reset()
            state = self.board.to_state()
            done = False
            while not done:
                # Wolf's turn
                # state, reward = self.wolf_turn_rl(state)
                state, reward = self.wolf_turn_random()

                # Sheep's turn
                state, reward = self.sheep_turn_rl(state)

                #print board
                # self.board.print()
                # input()

                if reward != ONGOING_GAME:
                    self.board.reset()
                    state = self.board.to_state()
                    if reward == WOLF_WON:
                        self.wolf_won += 1
                    elif reward == SHEEP_WON:
                        self.sheep_won += 1
                    done = True

            self.wolf_epsilon = max(self.wolf_epsilon * self.wolf_epsilon_decay, self.wolf_epsilon_min)
            self.sheep_epsilon = max(self.sheep_epsilon * self.sheep_epsilon_decay, self.sheep_epsilon_min)

            if episode % save_interval == 0:
                self.save_models(wolf_path, sheep_path)
                print(f'Episode {episode}')
                print(f'Wolf Epsilon: {self.wolf_epsilon}')
                print(f'Sheep Epsilon: {self.sheep_epsilon}')
                print(f'Wolf wins: {self.wolf_won}')
                print(f'Sheep wins: {self.sheep_won}')


env = Environment()
env.load_models('garb.pt', 'rand_sheep.pt')
env.free_train(2000, 30000, 'garb.pt', 'rand_sheep.pt')
env.save_models('garb.pt', 'rand_sheep.pt')
# env.load_models()
# env.play()
