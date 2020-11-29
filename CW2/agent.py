############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import torch
from matplotlib import pyplot as plt
import collections

class Agent:

    # Function to initialise the agent
    def __init__(self, gamma=0.9, use_double_dqn=True, lr=0.001,
                 buffer_maxlen=5000, epsilon=1, epsilon_decay=0.99,
                 episode_length=1000, batch_size=100, update_target_network_period=20,
                 inspect_greedy_policy_period = 30):

        # Construct a DQN
        self.dqn = DQN(gamma=gamma, use_double_dqn=use_double_dqn, lr=lr)
        # Create a Replay Buffer
        self.buffer = ReplayBuffer(maxlen=buffer_maxlen)
        # Discrete action space
        self.discrete_action_space = list(range(4))  # [up, down, left, right]
        # self.discrete_action_space = list(range(8))  # [up, down, left, right, ne, nw, sw, se]
        # Number of discrete actions
        self.action_size = len(self.discrete_action_space)
        # Epsilon and epsilon decay for the epsilon-greedy policy
        self.epsilon = epsilon
        # self.epsilon_decay = 0.9999965
        self.epsilon_decay = epsilon_decay

        # Set the episode length
        # self.episode_length = 500
        self.episode_length = episode_length
        # Define the batch size
        self.batch_size = batch_size
        # Number of epochs to update the target network
        self.update_target_network_period = update_target_network_period
        # Number of epochs to inspect the greedy policy and update epsilon if needed
        self.inspect_greedy_policy_period = inspect_greedy_policy_period
        # Flag to determine whether to inspect the greedy policy
        self.inspect_greedy_policy = False
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # Minimum distance to goal
        self.min_distance_to_goal = 100 # Initialised with a high value

        # Create lists to store the losses and epochs
        self.losses = []
        self.iterations = []
        # List of losses in each step
        self.loss_list = []
        # Count number of epochs
        self.epoch_counter = 0

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            # Increment epoch_counter
            self.epoch_counter += 1
            if self.inspect_greedy_policy:
                print('Inspecting a greedy policy')
            else:
                # Store average loss in each episode
                if len(self.buffer.buffer) >= self.batch_size:
                    # Print out this loss
                    print('Iteration ' + str(self.epoch_counter) + ', Loss = ' + str(np.mean(self.loss_list)) + ', Epsilon = ' + str(self.epsilon))
                    # Store this loss in the list
                    self.losses.append(np.mean(self.loss_list))
                    # Update the list of iterations
                    self.iterations.append(self.epoch_counter)
            # Update the target network
            if self.epoch_counter % self.update_target_network_period == 0:
                self.dqn.update_target_network()
            # Inspect the greedy policy
            if self.epoch_counter % self.inspect_greedy_policy_period == 0:
                self.inspect_greedy_policy = True
                # Shorten the episode length when inspecting the greedy policy
                self.episode_length = 100
                self.min_distance_to_goal = 100  # Reset to a high value
            # Update epsilon based on the inspection of the greedy policy
            if self.epoch_counter % self.inspect_greedy_policy_period == 1:
                # Reset flag to False
                print('Epsilon Before: {}'.format(str(self.epsilon)))
                self.inspect_greedy_policy = False
                # Reset the episode length
                # self.episode_length = 500
                self.episode_length = 700
                new_epsilon = self.epsilon + 0.3 * (self.min_distance_to_goal**2)
                print('Min Distance to Goal: {}'.format(self.min_distance_to_goal))
                self.update_epsilon(specified_epsilon=new_epsilon)
                print('Epsilon After: {}'.format(str(self.epsilon)))
            # Re-initialise
            self.loss_list = []
            # Update epsilon
            self.update_epsilon()
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        if self.inspect_greedy_policy:
            return self.get_greedy_action(state)
        else:
            # Store the state; this will be used later, when storing the transition
            self.state = state
            # Pick a discrete action from an epsilon-greedy policy
            # Choose the next action.
            discrete_action = self._choose_next_action()
            # Convert the discrete action into a continuous action.
            continuous_action = self._discrete_action_to_continuous(discrete_action)
            # Store the action; this will be used later, when storing the transition
            self.action = discrete_action
            return continuous_action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Save the minimum distance to goal when inspecting the greedy policy
        if self.inspect_greedy_policy:
            if distance_to_goal < self.min_distance_to_goal:
                self.min_distance_to_goal = distance_to_goal
        # Distance between current and next states
        transition_distance = np.linalg.norm(next_state - self.state)
        # Penalty when hitting the wall
        if transition_distance == 0:
            hitting_wall_penalty = 0.1
        else:
            hitting_wall_penalty = 0
        # Convert the distance to a reward
        reward = 1 - distance_to_goal - hitting_wall_penalty #+ (transition_distance * 2.5)
        # Create a transition
        transition = (self.state, self.action, reward, next_state)

        # Only save transitions and train the model when not inspecting the greedy policy
        if not self.inspect_greedy_policy:
            # Save transition to the buffer
            self.buffer.append(transition)
            # Wait until the buffer size >= batch size
            if len(self.buffer.buffer) >= self.batch_size:
                # Sample a random mini batch
                mini_batch = self.buffer.sample_mini_batch(self.batch_size)
                # Train the network and obtain the scalar loss
                loss_value = self.dqn.train_q_network(mini_batch)
                self.loss_list.append(loss_value)
                # Update the weights for the prioritised experience replay buffer
                individual_loss_list = self.dqn.calculate_loss_per_transition(mini_batch)
                self.buffer.update_weight(individual_loss_list)
    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # Get Q values from q_network
        q_vals = self.dqn.q_network(torch.tensor(state))
        q_vals = q_vals.detach().numpy()

        # Find index of the action with the highest Q value
        best_action = np.argmax(q_vals)

        specified_epsilon = 0 # Greedy policy
        epsilon_greedy_policy = self._generate_policy(best_action, specified_epsilon)
        # Choose the next action.
        discrete_action = np.random.choice(self.discrete_action_space, p=epsilon_greedy_policy)
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)

        return continuous_action

    def update_epsilon(self, specified_epsilon=None):
        if specified_epsilon is None:
            # Update epsilon as usual
            self.epsilon = self.epsilon * self.epsilon_decay
        else:
            # Use the specified epsilon value
            self.epsilon = specified_epsilon
            # Make sure epsilon is not higher than 1
            if self.epsilon > 1:
                self.epsilon = 1

    # Function for the agent to choose its next action
    def _choose_next_action(self):
        # Get Q values from q_network
        q_vals = self.dqn.q_network(torch.tensor(self.state))
        q_vals = q_vals.detach().numpy()

        # Find index of the action with the highest Q value
        best_action = np.argmax(q_vals)

        epsilon_greedy_policy = self._generate_policy(best_action)

        return np.random.choice(self.discrete_action_space, p=epsilon_greedy_policy)

    def _generate_policy(self, best_action, specified_epsilon=None):

        if specified_epsilon is None:
            current_epsilon = self.epsilon
        else:
            current_epsilon = specified_epsilon

        greedy_action_prob = 1 - current_epsilon + (current_epsilon / self.action_size)
        non_greedy_action_prob = current_epsilon / self.action_size

        policy = np.zeros(self.action_size)
        policy += non_greedy_action_prob
        policy[best_action] = greedy_action_prob

        return policy

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        ##
        discrete_to_continuous_mappings = {0: np.array([0, 0.01], dtype=np.float32), # up
                                           1: np.array([0, -0.01], dtype=np.float32), # down
                                           2: np.array([-0.01, 0], dtype=np.float32), # left
                                           3: np.array([0.01, 0], dtype=np.float32)} # right
        # discrete_to_continuous_mappings = {0: np.array([0, 0.01], dtype=np.float32), # up
        #                                    1: np.array([0, -0.01], dtype=np.float32), # down
        #                                    2: np.array([-0.01, 0], dtype=np.float32), # left
        #                                    3: np.array([0.01, 0], dtype=np.float32), # right
        #                                    4: np.array([0.01/np.sqrt(2), 0.01/np.sqrt(2)], dtype=np.float32), # north east
        #                                    5: np.array([-0.01/np.sqrt(2), 0.01/np.sqrt(2)], dtype=np.float32), # north wast
        #                                    6: np.array([-0.01/np.sqrt(2), -0.01/np.sqrt(2)], dtype=np.float32), # south west
        #                                    7: np.array([0.01 / np.sqrt(2), -0.01 / np.sqrt(2)], dtype=np.float32),} # south east
        return discrete_to_continuous_mappings[discrete_action]

    # Plot the loss curve
    def plot_loss(self, figure_name="figures/loss_vs_iterations.png"):
        # Create a graph which will show the loss as a function of the number of training iterations
        fig, ax = plt.subplots()
        ax.set(xlabel='Iteration', ylabel='Loss')
        ax.plot(self.iterations, self.losses, color='blue')
        plt.yscale('log')
        plt.grid()
        fig.savefig(figure_name)

# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self, gamma=0.9, use_double_dqn=False, lr=0.001):
        # Discount rate used in the Bellman Equation
        self.gamma = gamma
        # Flag to determine whether to use Double Deep Q-Learning or not
        self.use_double_dqn = use_double_dqn
        # Create a new Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Create a target network
        self.target_network = Network(input_dimension=2, output_dimension=4)
        # Initialise the weights of the target network to be the same as that of self.q_network
        self.update_target_network()
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=lr)

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Update the target network
    def update_target_network(self):
        network_state_dict = self.q_network.state_dict()

        self.target_network.load_state_dict(network_state_dict)

    # Calculate loss per transition in a batch for prioritised replay buffer
    def calculate_loss_per_transition(self, transition):
        if self.use_double_dqn:
            expected_returns, pred_q = self._calculate_expected_and_predicted_q_double_dqn(transition)
        else:
            expected_returns, pred_q = self._calculate_expected_and_predicted_q(transition)

        individual_loss_list = [(i - j)**2 for i, j in zip(expected_returns.detach().numpy(), pred_q.detach().numpy())]

        return individual_loss_list

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transition):
        if self.use_double_dqn:
            expected_returns, pred_q = self._calculate_expected_and_predicted_q_double_dqn(transition)
        else:
            expected_returns, pred_q = self._calculate_expected_and_predicted_q(transition)

        loss = torch.nn.MSELoss()(expected_returns, pred_q)

        return loss

    # Calculate the expected returns and the predicted q values using Deep Q-Learning
    def _calculate_expected_and_predicted_q(self, transition):

        states, actions, rewards, next_states = self._unpack_transition(transition)

        # Get Q values from q_network given current states
        pred_q = self.q_network(states)
        pred_q = pred_q.gather(dim=1, index=actions.unsqueeze(-1)).squeeze(-1)

        # The actual immediate reward
        actual_r = rewards

        # Calculate Q values for the next states
        next_pred_q = self.target_network(next_states) # Use target network
        # next_pred_q = self.q_network(next_states)  # When not using target network
        next_best_actions = next_pred_q.argmax(dim=1)
        next_best_pred_q = next_pred_q.gather(dim=1, index=next_best_actions.unsqueeze(-1)).squeeze(-1)

        # Calculate expected discounted sum of future rewards (returns)
        expected_returns = actual_r + self.gamma * next_best_pred_q

        return expected_returns, pred_q

    # Calculate the expected returns and the predicted q values using Double Deep Q-Learning
    def _calculate_expected_and_predicted_q_double_dqn(self, transition):

        states, actions, rewards, next_states = self._unpack_transition(transition)

        # Get Q values from q_network given current states
        pred_q = self.q_network(states)
        pred_q = pred_q.gather(dim=1, index=actions.unsqueeze(-1)).squeeze(-1)

        # The actual immediate reward
        actual_r = rewards

        # Calculate Q values for the next states
        next_target_pred_q = self.target_network(next_states)  # Use target network
        next_target_best_actions = next_target_pred_q.argmax(dim=1) # Get next best actions from the target network

        # Get Q values from q_network given next states and target next best actions
        next_pred_q = self.q_network(next_states)
        next_best_pred_q = next_pred_q.gather(dim=1, index=next_target_best_actions.unsqueeze(-1)).squeeze(-1)

        # Calculate expected discounted sum of future rewards (returns)
        expected_returns = actual_r + self.gamma * next_best_pred_q

        return expected_returns, pred_q

    # Extract states, actions, rewards and next states separately
    # from transitions list and format them into tensors
    def _unpack_transition(self, transition):

        states = [tup[0] for tup in transition]
        actions = [tup[1] for tup in transition]
        rewards = [tup[2] for tup in transition]
        next_states = [tup[3] for tup in transition]

        return torch.tensor(states), torch.tensor(actions), torch.tensor(rewards, dtype=torch.float), torch.tensor(next_states)

class ReplayBuffer:

    def __init__(self, maxlen=5000):
        self.buffer = collections.deque(maxlen=maxlen)
        self.weights = collections.deque(maxlen=maxlen)

        # Indices of the randomly chosen batch
        self.batch_indices = None
        # Small weight constant
        self.weight_constant = 1e-3
        # Degree of prioritisation
        self.alpha = 2

    def append(self, transition):
        # Append the transition to the replay buffer
        self.buffer.append(transition)

        # Calculate the weight of the newly added transition
        self.assign_weight()

    def sample_mini_batch(self, batch_size):

        # Calculate sampling probabilities
        sampling_probs = self._calculate_sampling_probs()

        # Randomly pick indices of transitions inside the buffer
        self.batch_indices = np.random.choice(np.arange(len(self.buffer)), batch_size, p=sampling_probs).tolist()
        self.batch_indices = [int(idx) for idx in self.batch_indices]

        # Use the randomly picked indices to get the corresponding transitions
        mini_batch = [self.buffer[idx] for idx in self.batch_indices]

        return mini_batch

    def assign_weight(self):
        # Assign weight for the newly added transition
        if len(self.weights) == 0: # When adding the first transition
            weight = self.weight_constant
            self.weights.append(weight)
        else:
            # Assign weight equal to max weight
            weight = np.max(self.weights)
            self.weights.append(weight)

    def update_weight(self, individual_loss_list):
        # Update the weights of the randomly chosen batch based on their losses
        for loss, batch_idx in zip(individual_loss_list, self.batch_indices):
            self.weights[batch_idx] = loss + self.weight_constant

    def _calculate_sampling_probs(self):
        # Calculate the sampling probability for each transition in the buffer
        sampling_probs = [weight ** self.alpha for weight in self.weights]
        sampling_probs = sampling_probs / np.sum(sampling_probs)

        return sampling_probs