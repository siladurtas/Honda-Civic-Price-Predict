import pandas as pd
import numpy as np
import random
import os
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42

def evaluate_features(X, y, indices):
    if not indices:
        return -np.inf  
    X_subset = X.iloc[:, indices]
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y, test_size=0.2, random_state=RANDOM_STATE
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model.score(X_test, y_test)
    except ValueError:
        return -np.inf  

def genetic_algorithm(X, y, iterations=30, pop_size=30, mutation_rate=0.1):
    n = X.shape[1]
    population = [[random.randint(0, 1) for _ in range(n)] for _ in range(pop_size)]

    for _ in range(iterations):
        scored = [(ind, evaluate_features(X, y, [i for i, bit in enumerate(ind) if bit])) for ind in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        next_gen = [ind for ind, _ in scored[:pop_size // 2]]

        while len(next_gen) < pop_size:
            p1, p2 = random.choices(next_gen, k=2)
            point = random.randint(1, n - 1)
            child = p1[:point] + p2[point:]
            if random.random() < mutation_rate:
                idx = random.randint(0, n - 1)
                child[idx] ^= 1
            next_gen.append(child)

        population = next_gen

    best = max(population, key=lambda ind: evaluate_features(X, y, [i for i, bit in enumerate(ind) if bit]))
    return [X.columns[i] for i, bit in enumerate(best) if bit]

def particle_swarm(X, y, iterations=30, swarm_size=20):
    n = X.shape[1]

    class Particle:
        def __init__(self):
            self.pos = [random.randint(0, 1) for _ in range(n)]
            self.vel = [random.uniform(-1, 1) for _ in range(n)]
            self.best_pos = self.pos[:]
            self.best_score = evaluate_features(X, y, [i for i, b in enumerate(self.pos) if b])

    swarm = [Particle() for _ in range(swarm_size)]
    global_best = max(swarm, key=lambda p: p.best_score).best_pos

    for _ in range(iterations):
        for p in swarm:
            for i in range(n):
                inertia = 0.5 * p.vel[i]
                cognitive = 1.5 * random.random() * (p.best_pos[i] - p.pos[i])
                social = 1.5 * random.random() * (global_best[i] - p.pos[i])
                p.vel[i] = inertia + cognitive + social
                p.pos[i] = 1 if random.random() < 1 / (1 + np.exp(-p.vel[i])) else 0

            score = evaluate_features(X, y, [i for i, b in enumerate(p.pos) if b])
            if score > p.best_score:
                p.best_score = score
                p.best_pos = p.pos[:]

        global_best = max(swarm, key=lambda p: p.best_score).best_pos

    return [X.columns[i] for i, bit in enumerate(global_best) if bit]

def ant_colony(X, y, iterations=30, n_ants=10, decay=0.1):
    n = X.shape[1]
    pheromone = np.ones(n)
    best_features = []
    best_score = -np.inf

    for _ in range(iterations):
        for _ in range(n_ants):
            prob = pheromone / pheromone.sum()
            selected = np.random.rand(n) < prob
            indices = [i for i, val in enumerate(selected) if val]
            score = evaluate_features(X, y, indices)
            if score > best_score:
                best_score = score
                best_features = indices
            pheromone[indices] += score
        pheromone *= (1 - decay)

    return [X.columns[i] for i in best_features]

def select_features(X, y):
    os.makedirs('outputs', exist_ok=True)
    start = time.time()

    features_ga = genetic_algorithm(X, y)
    features_pso = particle_swarm(X, y)
    features_aco = ant_colony(X, y)

    all_features = features_ga + features_pso + features_aco
    voted = pd.Series(all_features).value_counts()
    final_features = voted[voted >= 2].index.tolist() 

    if not final_features:
        final_features = features_ga  

    visualize_feature_selection_results(
        features_ga, features_pso, features_aco, final_features, X.columns
    )

    with open('outputs/SelectedFeatures.txt', 'w') as f:
        f.writelines(f + '\n' for f in final_features)

    with open('outputs/feature_selection_time.txt', 'w') as f:
        f.write(f"Feature Selection Time: {time.time() - start:.2f} seconds\n")

    print("Final selected features saved and visualized successfully.")
    return final_features

def visualize_feature_selection_results(features_ga, features_pso, features_aco, final_features, all_features):
    results = pd.DataFrame(index=all_features)
    results['GA'] = [1 if f in features_ga else 0 for f in all_features]
    results['PSO'] = [1 if f in features_pso else 0 for f in all_features]
    results['ACO'] = [1 if f in features_aco else 0 for f in all_features]
    results['Ensemble'] = [1 if f in final_features else 0 for f in all_features]
    results['Total_Votes'] = results[['GA', 'PSO', 'ACO']].sum(axis=1)
    results = results.sort_values('Total_Votes', ascending=False)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    sns.heatmap(results[['GA', 'PSO', 'ACO', 'Ensemble']], 
                cmap='YlOrRd', 
                cbar=False,
                yticklabels=True)
    plt.title('Feature Selection Results by Algorithm')
    plt.xlabel('Selection Method')
    plt.ylabel('Features')

    plt.subplot(2, 1, 2)
    results['Total_Votes'].plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Total Votes per Feature')
    plt.xlabel('Features')
    plt.ylabel('Number of Algorithms Selected')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('outputs/feature_selection_visualization.png')
    plt.close()

    results.to_csv('outputs/feature_selection_results.csv')

    print("Feature selection visualization saved to 'outputs/feature_selection_visualization.png'")
    print("Detailed selection results saved to 'outputs/feature_selection_results.csv'")
