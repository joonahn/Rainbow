# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import torch
import collections


Transition_dtype = np.dtype([('timestep', np.int32), ('state', np.uint8, (7, 84, 84)), ('action', np.int32, (7,)), ('reward', np.float32, (7,)),
 ('nonterminal', np.bool_, (7,))])
blank_trans = (0, np.zeros((84, 84), dtype=np.uint8), np.zeros((7,), dtype=np.uint8), np.zeros((7,), dtype=np.float32), np.zeros((7,), dtype=np.bool_))


# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.tree_start = 2**(size-1).bit_length()-1  # Put all used node leaves on last tree level
        self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
        self.data = np.array([blank_trans] * size, dtype=Transition_dtype)  # Build structured array
        self.max = 1  # Initial max value to return (1 = 1^ω)
        self.min_epi_reward = 0.0

    # Updates nodes values from current tree
    def _update_nodes(self, indices):
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

    # Propagates changes up tree given tree indices
    def _propagate(self, indices):
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    # Propagates single value up tree given a tree index for efficiency
    def _propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate_index(parent)

    # Updates values given tree indices
    def update(self, indices, values):
        self.sum_tree[indices] = values  # Set new values
        self._propagate(indices)  # Propagate values
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)

    # Updates values given tree indices
    def update_reward(self, epi_id, epi_reward):
        # if target epi_id does not exists, return
        if self.data[self.data['epi_id'] == epi_id].size == 0:
            return
        self.min_epi_reward = min(self.min_epi_reward, epi_reward)
        delta_epi_reward = epi_reward - self.data[self.data['epi_id'] == epi_id]['epi_reward'][0]
        value = max(0.1, delta_epi_reward) #TODO: lower bound of epi reward is a hyperparam! 
        self.data[self.data['epi_id'] == epi_id]['epi_reward'] = epi_reward

        for index in np.arange(self.data.size)[self.data['epi_id'] == epi_id]:
            self._update_index(index + self.tree_start, value)

    # Updates values given tree indices
    def update_reward_by_index(self, data_index, epi_reward):
        # if target epi_id does not exists, return
        self.min_epi_reward = min(self.min_epi_reward, epi_reward)
        delta_epi_reward = epi_reward - self.data[data_index]['epi_reward']
        value = max(0.1, delta_epi_reward) #TODO: lower bound of epi reward is a hyperparam! 

        self.data[data_index]['epi_reward'] = epi_reward
        self._update_index(data_index + self.tree_start, value)

    # Updates values given tree indices
    def update_value_by_index(self, data_index, value):
        # if target epi_id does not exists, return
        self._update_index(data_index + self.tree_start, value)

    # Updates single value given a tree index for efficiency
    def _update_index(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate_index(index)  # Propagate value
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data  # Store data in underlying data structure
        self._update_index(self.index + self.tree_start, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    # Searches for the location of values in sum tree
    def _retrieve(self, indices, values):
        children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1)) # Make matrix of children indices
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(np.int32)  # Classify which values are in left or right branches
        successor_indices = children_indices[successor_choices, np.arange(indices.size)] # Use classification to index into the indices matrix
        successor_values = values - successor_choices * left_children_values  # Subtract the left branch values when searching in the right branch
        return self._retrieve(successor_indices, successor_values)

    # Searches for the location of values in sum tree
    def _retrieve_mod(self, indices, values):
        update_targets = (indices < self.tree_start)
        if (~update_targets).all():
            return indices
        children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1)) # Make matrix of children indices
        left_children_values = values.copy()
        left_children_values[update_targets] = self.sum_tree[children_indices[0][update_targets]]
        successor_choices = np.greater(values, left_children_values).astype(np.int32)  # Classify which values are in left or right branches
        successor_indices = indices.copy()
        successor_indices[update_targets] = children_indices[successor_choices, np.arange(indices.size)][update_targets] # Use classification to index into the indices matrix
        successor_values = values - successor_choices * left_children_values * update_targets.astype(np.int32)  # Subtract the left branch values when searching in the right branch
        return self._retrieve_mod(successor_indices, successor_values)

    # Searches for the location of values in sum tree
    def _retrieve_one_by_one(self, indices, values):
        children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1)) # Make matrix of children indices
        if indices[0] >= self.tree_start:
            return indices
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(np.int32)  # Classify which values are in left or right branches
        successor_indices = children_indices[successor_choices, np.arange(indices.size)] # Use classification to index into the indices matrix
        successor_values = values - successor_choices * left_children_values  # Subtract the left branch values when searching in the right branch
        return self._retrieve_one_by_one(successor_indices, successor_values)

    # Searches for values in sum tree and returns values, data indices and tree indices
    def find(self, values):
        indices = self._retrieve_mod(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return (self.sum_tree[indices], data_index, indices)  # Return values, data indices, tree indices

    def find_one_by_one(self, values):
        indices = []
        for value in values:
            index = self._retrieve_one_by_one(np.zeros([1], dtype=np.int32), np.array([value]))
            indices.append(index[0])
        indices = np.array(indices)
        data_index = indices - self.tree_start
        return (self.sum_tree[indices], data_index, indices)  # Return values, data indices, tree indices

    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]

    def evict_by_indices(self, indices):
        for index in indices: # tree-index
            data_index = index - self.tree_start
            self.index = (self.index - 1) % self.size # data-index
            self.data[data_index % self.size] = self.data[self.index]
            self.data[self.index] = blank_trans
            self._update_index(index, self.sum_tree[self.index + self.tree_start])
            self._update_index(self.index + self.tree_start, 0.0)

    def evict(self, count):
        value = self.sum_tree[self.tree_start:]
        zero_mask = (value == 0.0).astype(np.float32) * 1e38
        evict_targets = np.argsort(value + zero_mask)[:count] + self.tree_start
        self.evict_by_indices(evict_targets)
        del zero_mask

    def get_n_lowest_transitions(self, count):
        value = self.sum_tree[self.tree_start:]
        zero_mask = (value == 0.0).astype(np.float32) * 1e38
        targets = np.argsort(value + zero_mask)[:count]
        del zero_mask
        return self.data[targets], value[targets]

    def get_n_highest_transitions(self, count):
        value = self.sum_tree[self.tree_start:]
        targets = np.argsort(-(value))[:count]
        return self.data[targets], value[targets]

class ReplayMemory():
    def __init__(self, args, capacity):
        self.device = args.device
        self.capacity = capacity
        self.history = args.history_length
        self.discount = args.discount
        self.n = args.multi_step
        self.priority_weight = args.priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_exponent = args.priority_exponent
        self.t = 0  # Internal episode timestep counter
        self.n_step_scaling = torch.tensor([self.discount ** i for i in range(self.n)], dtype=torch.float32, device=self.device)  # Discount-scaling vector for n-step returns
        self.transitions = SegmentTree(capacity)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities
        self.buffer = collections.deque()
        self.current_idx = 0
        self.partition_size = args.partition_size
        self.partition_cnt = capacity // self.partition_size
        self.prev_dist = np.zeros((self.partition_cnt,))

    def append(self, state, action, reward, terminal):
        self.buffer.append((state, action, reward, terminal))
        if len(self.buffer) >= 7:
            states = np.array([v[0][-1].mul(255).to(dtype=torch.uint8, device=torch.device('cpu')).numpy() for v in self.buffer])
            actions = np.array([v[1] for v in self.buffer])
            rewards = np.array([v[2] for v in self.buffer])
            terminals = np.array([v[3] for v in self.buffer])
            self.transitions.append((self.t, states, actions, rewards, ~ terminals), self.transitions.max)  # Store new transition with maximum priority
            self.t = 0 if terminals[3] else self.t + 1  # Start new episodes with t = 0
            self.buffer.popleft()

    # Returns the transitions with blank states where appropriate
    def _get_transitions(self, idxs):
        transition_idxs = idxs
        transitions = self.transitions.get(transition_idxs)
        transitions_firsts = transitions['timestep'] == 0
        blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
        for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
            blank_mask[t] = np.logical_or(blank_mask[t + 1], transitions_firsts[t + 1]) # True if future frame has timestep 0
        for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
            blank_mask[t] = np.logical_or(blank_mask[t - 1], transitions_firsts[t]) # True if current or past frame has timestep 0
        transitions[blank_mask] = blank_trans
        return transitions

    # Returns a valid sample from each segment
    def _get_samples_from_segments(self, batch_size, p_total):
        segment_length = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
        segment_starts = np.arange(batch_size) * segment_length
        valid = False
        while not valid:
            samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts  # Uniformly sample from within all segments
            probs, idxs, tree_idxs = self.transitions.find(samples)  # Retrieve samples from tree with un-normalised probability
            if np.all((self.transitions.index - idxs) % self.capacity > self.n) and np.all((idxs - self.transitions.index) % self.capacity >= self.history) and np.all(probs != 0):
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0
        # Retrieve all required transition data (from t - h to t + n)
        transitions = self._get_transitions(idxs)
        # Create un-discretised states and nth next states
        all_states = transitions['state']
        states = torch.tensor(all_states[:, :self.history], device=self.device, dtype=torch.float32).div_(255)
        next_states = torch.tensor(all_states[:, self.n:self.n + self.history], device=self.device, dtype=torch.float32).div_(255)
        # Discrete actions to be used as index
        actions = torch.tensor(np.copy(transitions['action'][:, self.history - 1]), dtype=torch.int64, device=self.device)
        # Calculate truncated n-step discounted returns R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
        rewards = torch.tensor(np.copy(transitions['reward'][:, self.history - 1:-1]), dtype=torch.float32, device=self.device)
        R = torch.matmul(rewards, self.n_step_scaling)
        # Mask for non-terminal nth next states
        nonterminals = torch.tensor(np.expand_dims(transitions['nonterminal'][:, self.history + self.n - 1], axis=1), dtype=torch.float32, device=self.device)
        return probs, idxs, tree_idxs, states, actions, R, next_states, nonterminals

    def sample(self, batch_size):
        p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
        probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = self._get_samples_from_segments(batch_size, p_total)  # Get batch of valid samples
        probs = probs / p_total  # Calculate normalised probabilities
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)  # Normalise by max importance-sampling weight from batch
        return tree_idxs, states, actions, returns, next_states, nonterminals, weights, idxs

    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        self.transitions.update(idxs, priorities)

    def get_sample_by_indices(self, idxs, set_size=32):
        data = self._get_transitions(idxs)
        data = data[:data.size // set_size * set_size]
        states = torch.tensor(data['state'][:,:self.history], dtype=torch.float32, device=self.device).div_(255)
        next_states = torch.tensor(data['state'][:,1:self.history+1], dtype=torch.float32, device=self.device).div_(255)
        # states = states.view(-1, set_size, 7056) # 84*84
        # states = states.permute(0, 2, 1)
        actions = torch.tensor(np.copy(data['action'][:, self.history - 1]), dtype=torch.float32, device=self.device)
        actions = actions.view(-1, 1, set_size)
        rewards = torch.tensor(np.copy(data['reward'][:, self.history - 1:-1]), dtype=torch.float32, device=self.device)
        rewards = torch.matmul(rewards, self.n_step_scaling).view(-1, 1, set_size)
        return states, actions, rewards, next_states

    # Set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0
        return self

    # Return valid states for validation
    def __next__(self):
        if self.current_idx == self.capacity:
            raise StopIteration
        transitions = self.transitions.data[np.arange(self.current_idx - self.history + 1, self.current_idx + 1)]
        transitions_firsts = transitions['timestep'] == 0
        blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
        for t in reversed(range(self.history - 1)):
            blank_mask[t] = np.logical_or(blank_mask[t + 1], transitions_firsts[t + 1]) # If future frame has timestep 0
        transitions[blank_mask] = blank_trans
        state = torch.tensor(transitions['state'][:, self.history], dtype=torch.float32, device=self.device).div_(255)  # Agent will turn into batch
        self.current_idx += 1
        return state

    def update_reward(self, epi_id, epi_reward):
        self.transitions.update_reward(epi_id, epi_reward)

    def update_reward_by_indices(self, data_indices, epi_reward):
        for data_index in data_indices:
            self.transitions.update_reward_by_index(data_index, epi_reward)

    def update_value_by_indices(self, data_indices, value):
        for data_index in data_indices:
            self.transitions.update_value_by_index(data_index, value)

    def evict(self, count):
        self.transitions.evict(count)

    def get_lowesthighest_images(self, count):
        lowest, low_val = self.transitions.get_n_lowest_transitions(count)
        highest, high_val = self.transitions.get_n_highest_transitions(count)
        lowest = lowest['state']
        highest = highest['state']
        return lowest, highest, low_val, high_val

    def calculate_kl_divergence(self):
        partition_cnt = self.capacity // self.partition_size


    next = __next__  # Alias __next__ for Python 2 compatibility
