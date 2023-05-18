import os
import sys
import torch
from CatieAgentEnv import CatieAgentEnv
from ppo import PPO
from network import ForwardNet
from eval_policy import eval_policy
import argparse


def get_args():
    """
        Description:
        Parses arguments at command line.
        Parameters:
        Return:
            args - the arguments parsed
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', dest='mode', type=str, default='train')  # can be 'train' or 'test'
    parser.add_argument('--name', dest='name', type=str, default='')  # name of network
    parser.add_argument('--actor', dest='actor_model', type=str, default='')  # your actor model filename
    parser.add_argument('--critic', dest='critic_model', type=str, default='')  # your critic model filename

    args = parser.parse_args()

    return args


def train(env, hyperparameters, actor_model, critic_model):
    """
        Trains the model.
        Parameters:
            env : the environment to train on
            hyperparameters : a dict of hyperparameters to use, defined in main
            actor_model : the actor model to load in if we want to continue training
            critic_model : the critic model to load in if we want to continue training
        Return:
            None
    """
    print("Training", flush=True)

    # Create a model for PPO.
    model = PPO(env=env,
                **hyperparameters)

    # Tries to load in an existing actor/critic model to continue training on
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print("Successfully loaded.", flush=True)
    # Don't train from scratch if user accidentally forgets actor/critic model
    elif actor_model != '' or critic_model != '':
        print("Error: Either specify both actor/critic models or none at all.", end=" ")
        print("We don't want to accidentally override anything!", flush=True)
        sys.exit(0)
    else:
        print("Training from scratch.", flush=True)

    # Train the PPO model with a specified total batches
    model.learn(n_batches=hyperparameters['n_batches'])


def test(env, hyperparameters, actor_model, num_iterations=100000):
    """
        Tests the model.
        Parameters:
            env - the environment to test the policy on
            actor_model - the actor model to load in
        Return:
            None
    """
    print(f"Testing {actor_model}", flush=True)

    # If the actor model is not specified, then exit
    if actor_model == '':
        print("Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    # Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    hidden_size = hyperparameters['hidden_size']
    hidden_layers = hyperparameters['hidden_layers']

    # Build our policy the same way we build our actor model in PPO
    policy = ForwardNet(obs_dim, hidden_layers, hidden_size, act_dim)

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))

    # Evaluate our policy with a separate module, eval_policy, to demonstrate
    # that once we are done training the model/policy with ppo.py, we no longer need
    # ppo.py since it only contains the training algorithm. The model/policy itself exists
    # independently as a binary file that can be loaded in with torch.
    # This call also plots the needed data.
    eval_policy(policy=policy, env=env, n_iterations=num_iterations, name=hyperparameters['name'])


def main(args):
    """
        The main function to run.
        Parameters:
            args - the arguments parsed from command line
        Return:
            None
    """
    # NOTE: Here's where you can set hyperparameters for PPO.
    hyperparameters = {
        'gamma': 1,
        'lr': 1e-4,
        'clip': 0.2,
        'hidden_size': 20,
        'hidden_layers': 4,
        'n_episodes': 512,  # Number of episodes per batch used for batch learning
        'n_repetitions': 4096,  # Number of repetitions for testing every few batches
        'n_trials': 100,  # Default for current experiment
        'n_batches': 100000,
        'name': args.name
    }

    # Creates the environment we'll be running. Makes sure environment is set up properly.
    env = CatieAgentEnv()
    # Train or test, depending on the mode specified
    if args.mode == 'train':
        if not os.path.exists(f"./{hyperparameters['name']}"):
            os.makedirs(f"./{hyperparameters['name']}")
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
    else:
        if not os.path.exists(f"Plots/{hyperparameters['name']}"):
            os.makedirs(f"Plots/{hyperparameters['name']}")
        test(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model)


if __name__ == '__main__':
    args_ = get_args()  # Parse arguments from command line
    main(args_)
