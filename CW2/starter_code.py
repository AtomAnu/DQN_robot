# Import some modules from other libraries
import numpy as np
import torch
import time
from matplotlib import pyplot as plt
import collections
import pickle

# Import the environment module
from environment import Environment

# Import the visualiser module
from q_value_visualiser import QValueVisualiser

# The Agent class allows the agent to interact with the environment.
class Agent:

    # The class initialisation function.
    def __init__(self, environment, dqn=None, epsilon=1, epsilon_decay=1):
        # Set the agent's environment.
        self.environment = environment
        if dqn is None:
            ## Construct a new DQN
            self.dqn = DQN()
        else:
            self.dqn = dqn
        ## Discrete action space
        self.discrete_action_space = list(range(4)) # [up, down, left, right]
        ## Number of discrete actions
        self.action_size = len(self.discrete_action_space)
        # Create the agent's current state
        self.state = None
        # Create the agent's total reward for the current episode.
        self.total_reward = None
        ## Epsilon and epsilon decay for the epsilon-greedy policy
        self.epsilon = epsilon
        # self.epsilon_decay = 0.9999965
        self.epsilon_decay = epsilon_decay
        # Reset the agent.
        self.reset()

    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.environment.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0

    # Function to make the agent take one step in the environment.
    def step(self):
        # Choose the next action.
        discrete_action = self._choose_next_action()
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this paction.
        reward = self._compute_reward(distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward

        # Return the transition
        return transition

    def update_epsilon(self):
        # Update epsilon
        self.epsilon = self.epsilon * self.epsilon_decay

    # Function for the agent to choose its next action
    def _choose_next_action(self):

        q_vals = self.dqn.q_network(torch.tensor(self.state))
        q_vals = q_vals.detach().numpy()

        best_action = np.argmax(q_vals)

        epsilon_greedy_policy = self._generate_policy(best_action)

        return np.random.choice(self.discrete_action_space, p=epsilon_greedy_policy)

    def _generate_policy(self, best_action):

        greedy_action_prob = 1 - self.epsilon + (self.epsilon / self.action_size)
        non_greedy_action_prob = self.epsilon / self.action_size

        policy = np.zeros(self.action_size)
        policy += non_greedy_action_prob
        policy[best_action] = greedy_action_prob

        return policy

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        ##
        discrete_to_continuous_mappings = {0: np.array([0, 0.1], dtype=np.float32), # up
                                           1: np.array([0, -0.1], dtype=np.float32), # down
                                           2: np.array([-0.1, 0], dtype=np.float32), # left
                                           3: np.array([0.1, 0], dtype=np.float32)} # right
        return discrete_to_continuous_mappings[discrete_action]

    # Function for the agent to compute its reward. In this example, the reward is based on the agent's distance to the goal after the agent takes an action.
    def _compute_reward(self, distance_to_goal):
        reward = float(0.1*(1 - distance_to_goal))
        return reward


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
    def __init__(self, q_network=None, use_target_network=False, use_full_bellman=False, gamma=0.9):
        # Flag to use a target network
        self.use_target_network = use_target_network
        # Flag to use the full Bellman Equation to calculate the loss
        self.use_full_bellman = use_full_bellman
        # Discount rate used in the Bellman Equation
        self.gamma = gamma
        if q_network is None:
            # Create a new Q-network, which predicts the q-value for a particular state.
            self.q_network = Network(input_dimension=2, output_dimension=4)
        else:
            self.q_network = q_network
        # Check whether to use a target network or not
        if self.use_target_network:
            self.target_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

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

    def save_network(self, file_name='trained_q_network.pickle'):
        """
        Utility function to save the trained q-network model in trained_q_network.pickle.
        """
        with open(file_name, 'wb') as target:
            pickle.dump(self.q_network, target)

    @staticmethod
    def load_network(file_name='trained_q_network.pickle'):
        """
        Utility function to load the trained q-network model in trained_q_network.pickle.
        """
        with open(file_name, 'rb') as target:
            trained_model = pickle.load(target)
        return trained_model

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transition):

        states = [tup[0] for tup in transition]
        actions = [tup[1] for tup in transition]
        rewards = [tup[2] for tup in transition]
        next_states = [tup[3] for tup in transition]

        pred_q = self.q_network(torch.tensor(states))
        pred_q = pred_q.gather(dim=1, index=torch.tensor(actions).unsqueeze(-1)).squeeze(-1)

        # The actual immediate reward
        actual_r = torch.tensor(rewards)

        if self.use_full_bellman: # Use the full Bellman Equation
            # Calculate Q values for the next states
            if self.use_target_network:
                next_pred_q = self.target_network(torch.tensor(next_states)) # Use target network
            else:
                next_pred_q = self.q_network(torch.tensor(next_states)) # When not using target network
            next_best_actions = next_pred_q.argmax(dim=1)
            next_best_pred_q = next_pred_q.gather(dim=1, index=next_best_actions.unsqueeze(-1)).squeeze(-1)

            # Calculate expected discounted sum of future rewards (returns)
            expected_returns = actual_r + self.gamma * next_best_pred_q
            loss = torch.nn.MSELoss()(pred_q, expected_returns)
        else:
            # Use only the actual immediate reward
            loss = torch.nn.MSELoss()(pred_q, actual_r)

        return loss

class ReplayBuffer:

    def __init__(self):
        self.buffer = collections.deque(maxlen=5000)

    def append(self, transition):
        self.buffer.append(transition)

    def sample_mini_batch(self, batch_size):

        random_indices = np.random.choice(np.arange(len(self.buffer)), batch_size).tolist()
        random_indices = [int(idx) for idx in random_indices]

        mini_batch = [self.buffer[idx] for idx in random_indices]

        return mini_batch


# Main entry point
if __name__ == "__main__":

    # Define number of episodes to train the agent
    num_of_episodes = 1000
    # Define the episode length
    episode_length = 100
    # Define the batch size
    batch_size = 100
    # Flag for using a target network
    use_target_network = True
    # use_target_network = False
    print('Use Target Network: {}'.format(str(use_target_network)))
    # Number of epochs to update the target network
    update_target_network = 10
    print('Number of epochs to update the target network: {}'.format(str(update_target_network)))
    # Count number of epochs until update_target_network
    epoch_counter = 0
    # Flag for using the full Bellman Equation
    use_full_bellman = True
    # use_full_bellman = False
    print('Use Full Bellman: {}'.format(str(use_full_bellman)))
    # Specify the discount rate (gamma)
    gamma = 0.9
    print('Gamma: {}'.format(str(gamma)))
    # Specify the starting value of epsilon for the epsilon-greedy policy
    # epsilon = 1
    epsilon = 0.8
    print('Epsilon: {}'.format(str(epsilon)))
    # Specify the value of epsilon decay
    # epsilon_decay = 1
    epsilon_decay = 0.9999965
    print('Epsilon Decay: {}'.format(str(epsilon_decay)))
    # Flag for using the replay buffer
    use_buffer = True
    # use_buffer = False
    print('Use Buffer: {}'.format(str(use_buffer)))

    # Create an environment.
    # If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
    # Magnification determines how big the window will be when displaying the environment on your monitor. For desktop monitors, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
    environment = Environment(display=True, magnification=500)
    # Create a DQN (Deep Q-Network)
    dqn = DQN(use_target_network=use_target_network, use_full_bellman=use_full_bellman, gamma=gamma)
    # Create an agent
    agent = Agent(environment=environment, dqn=dqn, epsilon=epsilon, epsilon_decay=epsilon_decay)
    ## Create a Replay Buffer
    buffer = ReplayBuffer()

    # Create lists to store the losses and epochs
    losses = []
    iterations = []

    # Create a graph which will show the loss as a function of the number of training iterations
    fig, ax = plt.subplots()
    ax.set(xlabel='Iteration', ylabel='Loss')

    # Loop over training iterations
    for training_iteration in range(num_of_episodes):
        # Reset the environment for the start of the episode.
        agent.reset()
        # List of losses in one episode
        loss_list = []
        # Loop over steps within this episode.
        for step_num in range(episode_length):
            # Step the agent once, and get the transition tuple for this step
            transition = agent.step()

            if use_buffer:
                ## Save transition to the buffer
                buffer.append(transition)
                if len(buffer.buffer) >= batch_size:
                    # Sample a random mini batch
                    mini_batch = buffer.sample_mini_batch(batch_size)
                    # Train the network and obtain the scalar loss
                    loss_value = agent.dqn.train_q_network(mini_batch)
                    loss_list.append(loss_value)
            else:
                # Train the network and obtain the scalar loss
                loss_value = agent.dqn.train_q_network([transition])
                loss_list.append(loss_value)
            # Sleep, so that you can observe the agent moving. Note: this line should be removed when you want to speed up training
            time.sleep(0.2)

        # Update epsilon
        agent.update_epsilon()

        if use_buffer:
            if len(buffer.buffer) >= batch_size:
                # Print out this loss
                print('Iteration ' + str(training_iteration) + ', Loss = ' + str(np.mean(loss_list)) + ', Epsilon = ' + str(agent.epsilon))
                # Store this loss in the list
                losses.append(np.mean(loss_list))
                # Update the list of iterations
                iterations.append(training_iteration)
        else:
            # Print out this loss
            print('Iteration ' + str(training_iteration) + ', Loss = ' + str(np.mean(loss_list)) + ', Epsilon = ' + str(
                agent.epsilon))
            # Store this loss in the list
            losses.append(np.mean(loss_list))
            # Update the list of iterations
            iterations.append(training_iteration)

        # Increment epoch_counter
        epoch_counter += 1
        if epoch_counter == update_target_network:
            epoch_counter = 0
            agent.dqn.update_target_network()

    # Save the trained q-network
    if not use_full_bellman and use_buffer:
        print('Saving the trained model')
        agent.dqn.save_network()

    # Save the trained q-network with the full implementation
    if use_target_network and use_full_bellman and use_buffer:
        print('Saving the trained model with the full implementation')
        agent.dqn.save_network(file_name='trained_q_network_full.pickle')

    # Flag to print the loss curve
    loss_curve = False

    if loss_curve:
        # Plot and save the loss vs iterations graph
        if use_buffer:
            if use_full_bellman:
                if use_target_network:
                    figure_name = "figures/loss_vs_iterations_target.png"
                else:
                    figure_name = "figures/loss_vs_iterations_bellman.png"
            else:
                figure_name = "figures/loss_vs_iterations_buffer.png"
        else:
            figure_name = "figures/loss_vs_iterations_online.png"
        ax.plot(iterations, losses, color='blue')
        plt.yscale('log')
        plt.grid()
        fig.savefig(figure_name)
