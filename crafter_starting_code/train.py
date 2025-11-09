import argparse  # to manage command-line arguments
import pickle  # to back-up evals statistics
from pathlib import Path  # to manage path

import torch  # for the deep learning

from src.crafter_wrapper import Env  # to prepare the environment of the game
from multiAgents import DQNAgent, DoubleDQNAgent

class RandomAgent:
    """An example Random Agent"""
##commentaire
    def __init__(self, action_num) -> None:
        self.action_num = action_num
        # a uniformly random policy
        self.policy = torch.distributions.Categorical(
            torch.ones(action_num) / action_num
        )

    def act(self, observation):
        """ Since this is a random agent the observation is not used."""
        return self.policy.sample().item()


def _save_stats(episodic_returns, crt_step, path):
    # save the evaluation stats
    episodic_returns = torch.tensor(episodic_returns)
    avg_return = episodic_returns.mean().item()
    print(
        "[{:06d}] eval results: R/ep={:03.2f}, std={:03.2f}.".format(
            crt_step, avg_return, episodic_returns.std().item()
        )
    )
    with open(path + "/eval_stats.pkl", "ab") as f:
        pickle.dump({"step": crt_step, "avg_return": avg_return}, f)


def eval(agent, env, crt_step, opt):
    """ Use the greedy, deterministic policy, not the epsilon-greedy policy you
    might use during training.
    """
    episodic_returns = []
    for _ in range(opt.eval_episodes):
        obs, done = env.reset(), False
        episodic_returns.append(0)
        while not done:
            action = agent.act(obs, epsilon=0.0)
            obs, reward, done, info = env.step(action)  # [FR] Effectue l'action. / [EN] Performs the action.
            episodic_returns[-1] += reward  # [FR] Ajoute la récompense au total de l'épisode. / [EN] Adds the reward to the episode's total.  episodic_returns[-1] += reward

    _save_stats(episodic_returns, crt_step, opt.logdir)


def _info(opt):
    try:
        int(opt.logdir.split("/")[-1])
    except:
        print(
            "Warning, logdir path should end in a number indicating a separate"
            + "training run, else the results might be overwritten."
        )
    if Path(opt.logdir).exists():
        print("Warning! Logdir path exists, results can be corrupted.")
    print(f"Saving results in {opt.logdir}.")
    print(
        f"Observations are of dims ({opt.history_length},84,84),"
        + "with values between 0 and 1."
    )


"""def main(opt):
    _info(opt)
    #opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.device = torch.device("cpu")
    env = Env("train", opt)
    eval_env = Env("eval", opt)
    #agent = RandomAgent(env.action_space.n)
    agent = DQNAgent(action_num=env.action_space.n)

    # main loop
    ep_cnt, step_cnt, done = 0, 0, True
    while step_cnt < opt.steps or not done:
        if done:
            ep_cnt += 1
            obs, done = env.reset(), False

        action = agent.act(obs)
        obs, reward, done, info = env.step(action)

        step_cnt += 1

        # evaluate once in a while
        if step_cnt % opt.eval_interval == 0:
            eval(agent, eval_env, step_cnt, opt)
"""


def main(opt):
    """[FR] La fonction principale qui exécute l'entraînement. / [EN] The main function that runs the training."""
    _info(opt)  # [FR] Affiche les informations de configuration. / [EN] Displays configuration information.
    opt.device = torch.device("cpu")  # [FR] Définit l'appareil sur CPU. / [EN] Sets the device to CPU.

    # [FR] Crée les environnements pour l'entraînement et l'évaluation.
    # [EN] Creates the environments for training and evaluation.
    env = Env("train", opt)
    eval_env = Env("eval", opt)

    agent_map = {
        'random': RandomAgent,
        'dqn': DQNAgent,
        'double_dqn': DoubleDQNAgent
    }

    # [FR] Crée l'agent spécifié par l'argument --agent.
    # [EN] Creates the agent specified by the --agent argument.
    if opt.agent == 'random':
        agent = RandomAgent(env.action_space.n)
    elif opt.agent in ['dqn', 'double_dqn']:
        AgentClass = agent_map[opt.agent]
        agent = AgentClass(
            action_num=env.action_space.n,
            history_length=opt.history_length,
            buffer_size=opt.buffer_size,
            batch_size=opt.batch_size,
            gamma=opt.gamma,
            learning_rate=opt.lr
        )
    else:
        raise ValueError(f"Agent inconnu: {opt.agent}. Choisissez parmi {list(agent_map.keys())}")

    print(f"Starting training for {opt.steps} steps with agent: {opt.agent}")
    step_cnt = 0
    obs = env.reset()
    while step_cnt < opt.steps:
        # [FR] Calculez epsilon (exemple de décroissance linéaire)
        # [EN] Calculate epsilon (example of linear decay)
        # epsilon = max(opt.epsilon_final, opt.epsilon_start - step_cnt / opt.epsilon_decay)

        # [FR] Utilisation d'un epsilon fixe pour l'instant
        # [EN] Using a fixed epsilon for now
        epsilon = 0.1

        # [FR] L'appel 'act' fonctionne maintenant pour tous les agents
        # [EN] The 'act' call now works for all agents
        action = agent.act(obs, epsilon=epsilon)

        next_obs, reward, done, info = env.step(action)

        # [FR] Logique d'apprentissage (uniquement pour les agents non-aléatoires)
        # [EN] Learning logic (only for non-random agents)
        if opt.agent != 'random':
            agent.replay_buffer.add(obs, action, reward, next_obs, done)
            agent.learn()

            if step_cnt % opt.target_update_interval == 0:
                agent.update_target_network()

        obs = next_obs
        step_cnt += 1

        if step_cnt % opt.eval_interval == 0:
            eval(agent, eval_env, step_cnt, opt)

        if done:
            obs = env.reset()

"""def get_options():
     Configures a parser. Extend this with all the best performing hyperparameters of
        your agent as defaults.

        For devel purposes feel free to change the number of training steps and
        the evaluation interval.
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logdir/random_agent/0")
    parser.add_argument(
        "--steps",
        type=int,
        metavar="STEPS",
        default=1_000_000,
        help="Total number of training steps.",
    )
    parser.add_argument(
        "-hist-len",
        "--history-length",
        default=4,
        type=int,
        help="The number of frames to stack when creating an observation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100_000,
        metavar="STEPS",
        help="Number of training steps between evaluations",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        metavar="N",
        help="Number of evaluation episodes to average over",
    )
    return parser.parse_args()
"""

def get_options():
    """[FR] Configure les arguments de la ligne de commande. / [EN] Configures the command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logdir/random_agent/0")
    parser.add_argument("--agent", type=str, default="random", choices=['random', 'dqn','double_dqn'], help="[FR] Type d'agent à entraîner. / [EN] Type of agent to train.")
    parser.add_argument("--steps", type=int, default=1_000_000, help="[FR] Nombre total de pas d'entraînement. / [EN] Total number of training steps.")
    parser.add_argument("--history-length", default=4, type=int, help="[FR] Nombre d'images à empiler pour une observation. / [EN] Number of frames to stack for an observation.")
    parser.add_argument("--eval-interval", type=int, default=100_000, help="[FR] Intervalle d'évaluation en pas. / [EN] Evaluation interval in steps.")
    parser.add_argument("--eval-episodes", type=int, default=20, help="[FR] Nombre d'épisodes pour l'évaluation. / [EN] Number of episodes for evaluation.")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Taille du Replay Buffer.")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille du lot pour l'apprentissage.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Facteur d'actualisation.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Taux d'apprentissage pour Adam.")
    parser.add_argument("--target-update-interval", type=int, default=1000,help="Fréquence de mise à jour du réseau cible.")

    return parser.parse_args()

if __name__ == "__main__":
    main(get_options())
