import torch
from collections import defaultdict
import time
import os
import numpy as np

from rl_games.common.player import BasePlayer
from rl_games.algos_torch import players
from rl_games.common.tr_helpers import unsqueeze_obs
from rl_games.algos_torch import torch_ext
import os

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action

class MTPlayer(BasePlayer):
    def __init__(self, params):
        # this is needed to invoke the vectorized environment
        params["config"]["player"] = params["config"].get(
            "player", 
            {"use_vecenv": True, "games_num": 1}
        )
        BasePlayer.__init__(self, params)
        
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0] 
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]

        obs_shape = self.obs_shape
        self._device = self.config.get('device', 'cuda:0')
        self.shuffle_data = self.config.get('shuffle_data', False)

        self.all_task_indices : torch.Tensor = self.env.env.extras["task_indices"]
        ordered_task_names : list[str] = self.env.env.extras["ordered_task_names"]

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)
        
        # this is the arguments for building the network
        build_config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : 1, # self.num_actors * self.num_agents, this is a dumy value
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
            'task_indices': self.all_task_indices,
            'task_embedding_dim': torch.unique(self.all_task_indices).shape[0],
            'ordered_task_names': ordered_task_names,
            'device': self._device
        }
        
        self.model = self.network.build(build_config)
        self.model.to(self.device)

        self.model.eval()
        self.is_rnn = self.model.is_rnn()  

    def restore(self, fn):
        self.base_folder = os.path.basename(fn).split(".")[0]
        self.base_folder = os.path.join("debug/mt_player", self.base_folder)
        self.video_recording_path = f"{self.base_folder}/videos"
        if not os.path.exists(self.video_recording_path):
            os.makedirs(self.video_recording_path)
        print(f"Loading checkpoint from {fn}")

        weights = torch.load(fn, map_location=self.device)
        # weights = torch.load(fn)
        self.set_weights(weights)

    def set_weights(self, weights):
        new_weight_dict = {}
        for k in weights['model']:
            if "value_mean_stds" in k:
                continue
            else:
                new_weight_dict[k] = weights['model'][k]
        self.model.load_state_dict(new_weight_dict, strict=False)
        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(
                weights['running_mean_std'])

    def restore(self, fn):
        if os.path.exists(fn):
            checkpoint = torch_ext.load_checkpoint(fn)
            if 0 in checkpoint:
                checkpoint = checkpoint[0]
            self.model.load_state_dict(checkpoint['model'])
            if self.normalize_input and 'running_mean_std' in checkpoint:
                self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

            env_state = checkpoint.get('env_state', None)
            if self.env is not None and env_state is not None:
                self.env.set_env_state(env_state)

        self.loaded_checkpoint = fn

    def get_action(self, obs, is_deterministic=False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
    
        processed_obs = self._preproc_obs(obs)
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'rnn_states' : self.states,
            'task_indices': self.all_task_indices
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action
        
    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        cumulative_rewards = defaultdict(list)
        success_rate = defaultdict(list)
        total_steps = defaultdict(list)
        games_played = defaultdict(int)
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            # print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        self.wait_for_checkpoint()

        need_init_rnn = self.is_rnn
        obses = self.env_reset(self.env)
        batch_size = 1
        batch_size = self.get_batch_size(obses, batch_size)

        if need_init_rnn:
            self.init_rnn()
            need_init_rnn = False

        cr = torch.zeros(batch_size, dtype=torch.float32)
        steps = torch.zeros(batch_size, dtype=torch.float32)
        global_steps = 0

        print_game_res = False

        for n in range(self.max_steps):
            # print(f"Step progress {n}", end='\r')
            if self.evaluation and n % self.update_checkpoint_freq == 0:
                self.maybe_load_new_checkpoint()

            if has_masks:
                masks = self.env.get_action_mask()
                action = self.get_masked_action(
                    obses, masks, is_deterministic)
            else:
                action = self.get_action(obses, is_deterministic)

            obses, r, done, info = self.env_step(self.env, action)
            cr += r
            steps += 1
            global_steps += 1

            if render:
                self.env.render(mode='human')
                time.sleep(self.render_sleep)

            for k in info:
                if "debug_visual" in k and info[k] is not None:
                    frames = info[k]  # (1, 200, 4, 240, 720)
                    # save frames into videos
                    import imageio
                    frames = frames.squeeze(0).detach().cpu().numpy()
                    frames = frames.transpose(0, 2, 3, 1)
                    imageio.mimsave(self.video_recording_path + f"/{k}_{n}.gif", frames, duration=100./3)
                    print(f"Saved video to {self.video_recording_path}/{k}_{n}.gif")

            all_done_indices = done.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
            done_count = len(done_indices)
            
            if done_count > 0:
                if self.is_rnn:
                    for s in self.states:
                        s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                cur_rewards = cr[done_indices]
                cur_steps = steps[done_indices]
                success = info["episode"]["success"]
                # import ipdb; ipdb.set_trace()

                cr = cr * (1.0 - done.float())
                steps = steps * (1.0 - done.float())
                
                # done_task_indices = self.all_task_indices[done_indices]
                # let each env be a unique task
                done_task_indices = torch.arange(done.shape[0])[done_indices]
                for i, task_index in enumerate(done_task_indices):
                    cumulative_rewards[task_index.item()].append(cur_rewards[i].item())
                    total_steps[task_index.item()].append(cur_steps[i].item())
                    success_rate[task_index.item()].append(success[i].item())
                    games_played[task_index.item()] += 1

                print(f'Overall success rate: {info["episode"]["average_environment_success_rate"]}')

                game_res = 0.0
                if isinstance(info, dict):
                    if 'battle_won' in info:
                        print_game_res = True
                        game_res = info.get('battle_won', 0.5)
                    if 'scores' in info:
                        print_game_res = True
                        game_res = info.get('scores', 0.5)
                
                min_game_played = min(games_played.values()) if len(games_played) > 0 else 0
                print(f"Game progress {min_game_played}; {len(success_rate)}/{self.all_task_indices.shape[0]}; Global steps: {global_steps}", end="\r")
                if min_game_played >= n_games and len(success_rate) == self.all_task_indices.shape[0]:
                    break

        # print game results
        # for task_index in cumulative_rewards.keys():
            # task_name = self.env.env.extras["ordered_task_names"][task_index]
        #     print(f"Task {task_index} \t\t- Mean reward: {sum(cumulative_rewards[task_index]) / len(cumulative_rewards[task_index])} \t Mean steps: {sum(total_steps[task_index]) / len(total_steps[task_index])} \t Success rate: {sum(success_rate[task_index]) / len(success_rate[task_index])}, Games played: {games_played[task_index]}")

        # compute average success rate
        success_rates = []
        for task_index in success_rate.keys():
            success_rates.append(success_rate[task_index][0])
        print(f"Average success rate: {sum(success_rates) / len(success_rates)}")

        # save success rate to file
        save_path = self.base_folder + "/success_rate.txt"
        print(f"Saving success rate to {save_path}")
        np.savetxt(save_path, success_rates, fmt="%.4f")
        overall_success_at_end_rate = 0
        for task_index in cumulative_rewards.keys():
            task_name = self.env.env.extras["ordered_task_names"][task_index]
            print(f"Task {task_name} \t\t- Mean reward: {sum(cumulative_rewards[task_index]) / len(cumulative_rewards[task_index])} \t Mean steps: {sum(total_steps[task_index]) / len(total_steps[task_index])} \t Success at end rate: {sum(success_rate[task_index]) / len(success_rate[task_index])}, Games played: {games_played[task_index]}")
            overall_success_at_end_rate += sum(success_rate[task_index]) / len(success_rate[task_index])
        
        print(f"Overall (success at end) rate: {overall_success_at_end_rate / len(cumulative_rewards.keys())}")
            
