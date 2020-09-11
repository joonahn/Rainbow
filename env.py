# -*- coding: utf-8 -*-
from collections import deque
import random
# import atari_py
# import cv2
import torch
import gym
import numpy as np


class Env():
    def __init__(self, args):
        self.device = args.device
        # self.ale = atari_py.ALEInterface()
        # self.ale.setInt('random_seed', args.seed)
        # self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        # self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        # self.ale.setInt('frame_skip', 0)
        # self.ale.setBool('color_averaging', False)
        # self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
        # actions = self.ale.getMinimalActionSet()
        # self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode
        self.env = gym.make("PongNoFrameskip-v4")

    def _get_state(self, state):
        # input state: numpy (160, 120, 3)
        state = torch.tensor(np.transpose(state, (2, 0, 1)), dtype=torch.float32, device=self.device)
        state = state.unsqueeze(0)
        state = torch.nn.functional.interpolate(state, size=(84, 84))
        state = state.squeeze(0)
        state = state[-1]
        # state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return state
        # return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            observation = self.env.step(0)
            # self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.env.reset()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(1, 30)):
                observation, _, done, _ = self.env.step(0)
                if done:
                    observation = self.env.reset()
        # Process and return "initial" state
        observation = self._get_state(observation)
        self.state_buffer.append(observation)
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            s, r, done, _ = self.env.step(action)
            reward += r

            # reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state(s)
            elif t == 3:
                frame_buffer[1] = self._get_state(s)
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return self.env.action_space.n
        # return len(self.actions)

    def render(self):
        # cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        # cv2.waitKey(1)
        pass

    def close(self):
        # cv2.destroyAllWindows()
        pass
