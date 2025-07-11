import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from copy import deepcopy
import matplotlib.pyplot as plt

data = pd.read_csv('/content/UK_Cities_New.csv')
data.columns = data.columns.str.strip()

city_col = 'City'
lat_col = 'Latitude'
lon_col = 'Longitude'
if lat_col not in data.columns or lon_col not in data.columns:
    raise KeyError("Expected columns 'Latitude' and 'Longitude' not found in CSV file.")

cities = data[[lat_col, lon_col]].values
city_names = data[city_col].values if city_col in data.columns else np.arange(len(cities))
num_cities = len(cities)

def euclidean_distance(coord1, coord2):
    return np.linalg.norm(coord1 - coord2)

distance_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(num_cities):
        distance_matrix[i, j] = euclidean_distance(cities[i], cities[j])


class TSPEnvironment:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.reset()

    def reset(self):
        self.visited = [np.random.randint(0, self.num_cities)]
        self.current_city = self.visited[0]
        self.mask = np.ones(self.num_cities, dtype=bool)
        self.mask[self.current_city] = False
        return self.current_city, self.mask.copy()

    def step(self, next_city):
        reward = -self.distance_matrix[self.current_city, next_city]
        self.visited.append(next_city)
        self.current_city = next_city
        self.mask[next_city] = False
        done = (len(self.visited) == self.num_cities)
        return (self.current_city, self.mask.copy()), reward, done

    def get_route_distance(self, route):
        dist = 0.0
        for i in range(len(route)-1):
            dist += self.distance_matrix[route[i], route[i+1]]

        dist += self.distance_matrix[route[-1], route[0]]
        return dist


class PPOPolicy(nn.Module):
    def __init__(self, num_cities, hidden_size=128):
        super(PPOPolicy, self).__init__()
        self.fc1 = nn.Linear(num_cities * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_cities)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        logits = self.fc2(x)
        return logits

def create_state_vector(current_city, mask, num_cities):
    one_hot = np.zeros(num_cities)
    one_hot[current_city] = 1.0
    state = np.concatenate([one_hot, mask.astype(float)])
    return state

def ppo_update(policy, optimizer, trajectories, clip_param=0.2, epochs=4):
    all_states = torch.tensor(np.array([t['state'] for t in trajectories]), dtype=torch.float32)
    all_actions = torch.tensor(np.array([t['action'] for t in trajectories]), dtype=torch.int64)
    all_rewards = torch.tensor(np.array([t['reward'] for t in trajectories]), dtype=torch.float32)
    old_log_probs = torch.tensor(np.array([t['log_prob'] for t in trajectories]), dtype=torch.float32)

    advantages = all_rewards

    for _ in range(epochs):
        logits = policy(all_states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(all_actions)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
        loss = -torch.min(surr1, surr2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


def run_policy(policy, env):
    current_city, mask = env.reset()
    route = [current_city]
    done = False
    while not done:
        state_vec = create_state_vector(current_city, mask, num_cities)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        logits = policy(state_tensor)
        logits_np = logits.detach().numpy().flatten()
        mask_float = mask.astype(float)
        masked_logits = torch.tensor(logits_np * mask_float - 1e8 * (1 - mask_float))
        dist_cat = torch.distributions.Categorical(logits=masked_logits)
        action = dist_cat.sample().item()
        (current_city, mask), _, done = env.step(action)
        route.append(action)
    return route, env.get_route_distance(route)


n_experiments = 3
total_episodes = 500
ppo_lr = 1e-3
n_policy_eval = 10


np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

results = []

for exp in range(n_experiments):

    policy = PPOPolicy(num_cities)
    optimizer = optim.AdamW(policy.parameters(), lr=ppo_lr, weight_decay=1e-4)

    conv_episodes = []
    conv_distances = []
    candidate_routes = []

    for episode in range(total_episodes):
        env = TSPEnvironment(distance_matrix)
        current_city, mask = env.reset()
        done = False
        route = [current_city]
        trajectory = []
        while not done:
            state_vec = create_state_vector(current_city, mask, num_cities)
            state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            logits = policy(state_tensor)
            logits_np = logits.detach().numpy().flatten()
            mask_float = mask.astype(float)
            masked_logits = torch.tensor(logits_np * mask_float - 1e8 * (1 - mask_float))
            dist_cat = torch.distributions.Categorical(logits=masked_logits)
            action = dist_cat.sample().item()
            log_prob = dist_cat.log_prob(torch.tensor(action)).item()
            (current_city, mask), reward, done = env.step(action)
            route.append(action)
            trajectory.append({'state': state_vec, 'action': action, 'reward': reward, 'log_prob': log_prob})

        loss = ppo_update(policy, optimizer, trajectory, clip_param=0.2, epochs=8)

        if episode % 50 == 0:
            route_distance = env.get_route_distance(route)
            conv_episodes.append(episode)
            conv_distances.append(route_distance)
            candidate_routes.append(deepcopy(route))
            print(f"Experiment {exp+1}, Episode {episode}: Distance = {route_distance:.2f}, Loss = {loss:.4f}")


    eval_distances = []
    for _ in range(n_policy_eval):
        env_eval = TSPEnvironment(distance_matrix)
        _, d = run_policy(policy, env_eval)
        eval_distances.append(d)
    avg_ppo = np.mean(eval_distances)


    random_dists = []
    for _ in range(n_policy_eval):
        route_rand = list(range(num_cities))
        random.shuffle(route_rand)
        env_rand = TSPEnvironment(distance_matrix)
        random_dists.append(env_rand.get_route_distance(route_rand))
    avg_random = np.mean(random_dists)


    best_idx = np.argmin(conv_distances)
    best_distance = conv_distances[best_idx]
    best_route = candidate_routes[best_idx]

    results.append({
        'experiment': exp+1,
        'conv_episodes': conv_episodes,
        'conv_distances': conv_distances,
        'best_distance': best_distance,
        'best_route': best_route,
        'avg_ppo': avg_ppo,
        'avg_random': avg_random
    })


table2 = pd.DataFrame({
    'Experiment': [r['experiment'] for r in results],
    'PPO Best Distance': [r['best_distance'] for r in results],

    'Improvement (%)': [100 * (r['avg_random'] - r['best_distance']) / r['avg_random'] for r in results]
})
print("\nTable 1: Optimal Solutions")
print(table2)


table3 = pd.DataFrame({
    'Experiment': [r['experiment'] for r in results],
    'PPO Average Distance': [r['avg_ppo'] for r in results],
    'Random Average Distance': [r['avg_random'] for r in results]
})
print("\nTable 2: Average Solutions")
print(table3)


plt.figure(figsize=(10, 6))
for r in results:
    plt.plot(r['conv_episodes'], r['conv_distances'], marker='o', label=f'Experiment {r["experiment"]}')
plt.xlabel("Episode")
plt.ylabel("Route Distance")
plt.title("Convergence Curves")
plt.legend()
plt.grid(True)
plt.show()

for r in results:
    best_route = r['best_route']
    best_route_complete = best_route + [best_route[0]]
    route_coords = cities[best_route_complete]

    plt.figure(figsize=(8, 6))
    plt.scatter(cities[:, 1], cities[:, 0], color='red', zorder=5)
    for i, (x, y) in enumerate(cities):
        plt.text(y, x, str(city_names[i]), fontsize=9, color='black')
    plt.plot(route_coords[:, 1], route_coords[:, 0], linestyle='-', color='green', linewidth=2)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Best Path for Experiment {r['experiment']} (Distance: {r['best_distance']:.2f})")
    plt.grid(True)
    plt.show()
