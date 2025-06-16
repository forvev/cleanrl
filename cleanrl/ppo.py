# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000 # for more complex envs such as atari you would need 1M for example
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128 # it means that for each policy rollout I will collect 4*128 data points for training
    # 128 is big enough to collect some meaningful info and small enough to update frequently
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99 # just a discount variable - 0.99 means that we should care about the future rewards
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2 # coef for determining how much the new policy can deviate from the old one
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01 # determines how much the agent should explore (the higher the value, the more exploration)
    """coefficient of the entropy"""
    vf_coef: float = 0.5 # coef for balaning the learning between the policy (actor) and the value (critic)
    """coefficient of the value function"""
    max_grad_norm: float = 0.5 # param for controling how much the gradients can change in a single update, improving the stability of the training
    """the maximum norm for the gradient clipping"""
    target_kl: float = None # is an optional safety check to keep policy updates from being too large.
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


# Since PPO deals with the vector env instead of gym env directly, we need this function
def make_env(env_id, idx, capture_video, run_name):
    if capture_video and idx == 0:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

# takes as an argument layer (used by PPO) and two initialization params
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std) # PPO uses orthogonal initialization on layer.weight
    torch.nn.init.constant_(layer.bias, bias_const) # PPO uses constant initialization on layer.weight
    return layer

# interacts with the environments 
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # critic NN
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Actor NN
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, s, action=None):
        logits = self.actor(s) # actor outputs logits which are unnomalized action probabilities
        # it means that they don't necessarily lie in <0,1> or sum up to 1
        # that's why we use Categorical which is basically a softmax operation to get the desired probability distribution
        probs = Categorical(logits=logits)
        # If no action is given, sample one stochastically from the current policy distribution:
        if action is None:
            action = probs.sample()
        # probs.log_prob(action) is basically log_{pi} (a_t|s_t). it answers: how confident was the policy in taking this action?”
        # probs.entropy() encourages exploration when high
        # self.critic(x): estimate the value function for the current state
        return action, probs.log_prob(action), probs.entropy(), self.critic(s)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps) # 512
    args.minibatch_size = int(args.batch_size // args.num_minibatches) # 128
    args.num_iterations = args.total_timesteps // args.batch_size # 500000 // 512
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # PPO collects data in parallel from multiple envs. Thay's why we use SyncVectorEnv
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    # the size of the array depends on the num_steps and num_envs, so in our case we will collect 512 data points
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0 # to track the number of env step
    start_time = time.time() # will help to calculate FPS
    next_obs, _ = envs.reset(seed=args.seed) # init the envs and get the first observation to start interacting with them
    next_obs = torch.Tensor(next_obs).to(device) # stores the initial observation
    next_done = torch.zeros(args.num_envs).to(device) # As the rollout proceeds, this array is updated every step to reflect which environments have completed an episode.

    for iteration in range(1, args.num_iterations + 1): # == one rollout
        # Annealing the rate if instructed to do so.
        # Annealing algorithm helps to explore. Or in different words, helps to escape the local optimum.
        if args.anneal_lr:
            # because of args.num_iterations it will lineary decreace to zero. So at the beginning it will be doing kinda brave escapes,
            # however it will stop after some steps.
            frac = 1.0 - (iteration - 1.0) / args.num_iterations # goes from 1 to 0
            lrnow = frac * args.learning_rate # decreases the learning rate linearly
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad(): # we just collect the data, we don't train - saves the memory
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1) # critic's prediction
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)): # walking backwards through the rollout
                if t == args.num_steps - 1: # last step reached
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                # how much better or worse the immediate outcome was compared with what the critic predicted for the current state.
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t] # one step TD error; how much better
                # or worse the outcome was compared to what the critic predicted for the current state.
                #
                # Generalized Advantage Estimation (GAE) is a way to compute advantages that reduces variance
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        # A complete pass over the rollout batch once, sliced into smaller minibatches
        # why?
        # sample efficiency!
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size): # 0, 512, 128
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                #  Re-computing logprobs and values with the updated policy
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                # we use log for the numerical stability (helpful for really small values)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp() #why? exp(log(a)-log(b)) = a/b. It can be 1.2, so it means that the new policy is better

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # On average, how surprised is the new policy when seeing actions from the old policy?
                    # there is a minus because it should be (-newlogprob + b_logprobs[mb_inds]) => old-new
                    old_approx_kl = (-logratio).mean() # approx_KL: E [-log(pi_new - pi_old)]
                    approx_kl = ((ratio - 1) - logratio).mean() # tells us how much the new policy differs from the old one
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()] # it tracks how ofter the PPO
                    # clipping mechanism is triggered (true or false), helping to monitor if the policy updates are frequently
                    # being limited by the clipping cooefficient.
                    # A high clip fraction suggests that I may want to lower the learning rate or increase the clipping coefficient,
                    # while a low clip fraction suggests that the policy updates are generally within the clipping range.

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    # 1e-8 is here just to avoid the division by zero
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss - this refers to the clipped objective
                pg_loss1 = -mb_advantages * ratio # standard policy gradient loss
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) # clipped policy gradient loss
                # which prevents the new policy from deviating too much from the old one
                pg_loss = torch.max(pg_loss1, pg_loss2).mean() # it does the max of negatives, however the paper
                # does the min of positives, so it's equivalent
                # pg_loss ensures that the policy update is not too large, which helps to stabilize training.

                # Value loss - to avoid overfitting
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2 #MSE between the predicted and the actual returns
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    ) # limits how much the new value prediction can change from the old one
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2 # MSE between the clipped value and the actual returns
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped) # ??
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef # combines the policy gradient loss,
                # entropy bonus and value loss into a single scalar loss for optimization

                # the learning part
                optimizer.zero_grad() # clear out the gradients
                loss.backward() # compute gradients
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step() # look at these gradients and decide how to update them

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # monitor how well the critic is learning
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
