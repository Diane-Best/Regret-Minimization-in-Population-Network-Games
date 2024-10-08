import os
import numpy as np
import networkx as nx
from collections import defaultdict
import sys
import matplotlib.pyplot as plt

class PopulationNetwork:
    def __init__(self, population_graph, V):
        self.population_graph = population_graph
        self.population_num = population_graph.number_of_nodes()
        self.V = V
        self.net = nx.Graph()
        self.population_nodes = defaultdict(list)
        self._build_network()

    def _build_network(self):
        for node_idx in range(self.V[-1]):
            for v_idx, node_num in enumerate(self.V[1:], start=1):
                if node_idx < node_num:
                    self.net.add_node(node_idx, v_idx=v_idx - 1)
                    self.population_nodes[v_idx - 1].append(node_idx)
                    break

        for begin_v, end_v in self.population_graph.edges:
            for begin_node in self.population_nodes[begin_v]:
                for end_node in self.population_nodes[begin_v]:
                    self.net.add_edge(begin_node, end_node)

class BaseGame:
    def __init__(self, population_network, Actions, M_u, M_v, lam=1):
        self.lam = lam  # Boltzmann temperature
        self.Actions = Actions
        self.M_u = M_u
        self.M_v = M_v
        self.action_num = len(Actions)
        self.population_num = population_network.population_num
        self.population_graph = population_network.population_graph
        self.population_nodes = population_network.population_nodes
        self.net = population_network.net
        self.V = population_network.V

    def _norm(self, a, b, r_min, r_max):
        return (r_max - r_min) * np.random.normal(a, b) + r_min

    def init_regret(self, init_meanR, init_stdR):
        regret = np.zeros((self.V[-1], self.action_num), dtype=np.float64)
        maxR1 = np.max(self.M_u) - np.min(self.M_u)
        minR1 = np.min(self.M_u) - np.max(self.M_u)

        for pid in self.population_graph.nodes:
            for node in self.population_nodes[pid]:
                regret[node] = [self._norm(init_meanR[pid][ai], init_stdR[pid][ai], minR1, maxR1)
                                for ai in range(self.action_num)]
        return regret

    def update_policy(self, regret):
        exp_regret = np.exp(self.lam * regret)
        return exp_regret / np.sum(exp_regret, axis=1, keepdims=True)

    def get_reward(self, policy, isVirtual=False):
        reward = np.zeros((self.V[-1], self.action_num), dtype=np.float64)
        for pid in self.population_graph.nodes:
            neipids = list(self.population_graph.neighbors(pid))
            num_neipids = len(neipids)

            for neipid in neipids:
                M_matrix = self.M_u if pid < neipid else self.M_v

                for node in self.population_nodes[pid]:
                    policy_neigh = policy[self.population_nodes[neipid]]
                    reward_matrix = np.dot(M_matrix, policy_neigh.T) if isVirtual else np.dot(
                        policy[node].reshape(-1, 1) * M_matrix, policy_neigh.T)
                    reward[node] += np.mean(reward_matrix, axis=1)

            for node in self.population_nodes[pid]:
                reward[node] /= num_neipids

        return reward

    def update_R(self, regret, policy, reward, t):
        virtual_rewards = self.get_reward(policy, isVirtual=True)
        regret += (virtual_rewards - reward[:, np.newaxis] - regret) / t

    def sim_algo(self, regret, T=1000):
        x_history, r_history = [], []
        policy = self.update_policy(regret)


        for t in range(1, T+1):
            ex = np.zeros((self.population_num, self.action_num), dtype=np.float64)
            er = np.zeros((self.population_num, self.action_num), dtype=np.float64)

            for pid in self.population_nodes:
                ex[pid] = [np.mean(policy[self.population_nodes[pid], ai]) for ai in range(self.action_num)]
                er[pid] = [np.mean(regret[self.population_nodes[pid], ai]) for ai in range(self.action_num)]

            x_history.append(ex)
            r_history.append(er)
            if t % 50 == 1:
                print(f'\r t: {t} / {T}; ex: {np.round(ex[:, 0], 2)}', end="")
            sys.stdout.flush()

            reward = np.sum(self.get_reward(policy, isVirtual=False), axis=1)
            self.update_R(regret, policy, reward, t)
            policy = self.update_policy(regret)

        print(f'\r t: {t} / {T}; ex: {np.round(ex[:, 0], 2)}', end="")

        return x_history, r_history

    def simulate(self, init_meanR, init_stdR, T=1000, repeat=5, save_tag=""):
        if save_tag:
            current_path = os.path.join(os.getcwd())
            src_path = os.path.join(current_path,'sim_'+save_tag+'_data_norm')
            if os.path.exists(src_path) and os.path.isdir(src_path):
                print(src_path)
                path_list = os.listdir(src_path)
                print("Filelist:")
                for file in path_list:
                    print(file)
            else:
                os.makedirs(src_path)
                print("Created Files.")
        for re in range(repeat):
            print(f"\n start: {re} / {repeat}")
            regret = self.init_regret(init_meanR, init_stdR)
            ExT, ErT = self.sim_algo(regret, T)
            if save_tag:
                os.makedirs(src_path, exist_ok=True)
                np.savez(f'{src_path}/sim_{save_tag}_data_norm_{re}.npz', init_r=regret, ExT=ExT, ErT=ErT)
        print("\nEnd!")

# Rock-Paper-Scissor (RPS) Game
class RPSGame(BaseGame):
    def __init__(self, population_network, lam=1):
        Actions = ['R', 'P', 'S']
        M_u = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=np.float64)
        M_v = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=np.float64)
        super().__init__(population_network, Actions, M_u, M_v, lam)


# Prisoner's Dilemma (PD) Game
class PDGame(BaseGame):
    def __init__(self, population_network, lam=1):
        Actions = ['C', 'D']
        M_u = np.array([[6, 2], [8, 2]], dtype=np.float64)
        M_v = np.array([[6, 2], [8, 2]], dtype=np.float64)
        super().__init__(population_network, Actions, M_u, M_v, lam)



if __name__ == '__main__':
    nodes_per_population = 100
    population_number = 20
    pop_graph = nx.watts_strogatz_graph(population_number, 4, 0.3)
    pop_network = PopulationNetwork(pop_graph, np.arange(0, nodes_per_population*population_number+1, nodes_per_population))
    game = PDGame(pop_network)

    rng_fixed_seed = np.random.default_rng(seed=42)
    init_meanR = rng_fixed_seed.random((pop_network.population_num, len(game.Actions)))
    init_stdR = np.full_like(init_meanR, 0.05)

    game.simulate(init_meanR, init_stdR, T=200, repeat=3, save_tag=f"ws_P{population_number}k4p0.3_pd")