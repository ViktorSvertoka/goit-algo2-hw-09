import random
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from colorama import Fore, Style, init

# Ініціалізація colorama
init(autoreset=True)


# Визначення функції Сфери
def sphere_function(x):
    return sum(xi**2 for xi in x)


# Hill Climbing
def hill_climbing(func, bounds, iterations=1000, epsilon=1e-6):
    dim = len(bounds)
    current = [random.uniform(*bounds[i]) for i in range(dim)]
    current_value = func(current)
    history = [current]

    for _ in range(iterations):
        neighbor = [current[i] + random.uniform(-epsilon, epsilon) for i in range(dim)]
        neighbor = [
            max(min(neighbor[i], bounds[i][1]), bounds[i][0]) for i in range(dim)
        ]
        neighbor_value = func(neighbor)

        if neighbor_value < current_value:
            current, current_value = neighbor, neighbor_value
            history.append(current)
        else:
            break

    return current, current_value, history


# Random Local Search
def random_local_search(func, bounds, iterations=1000, epsilon=1e-6):
    dim = len(bounds)
    best = [random.uniform(*bounds[i]) for i in range(dim)]
    best_value = func(best)
    history = [best]

    for _ in range(iterations):
        candidate = [random.uniform(*bounds[i]) for i in range(dim)]
        candidate_value = func(candidate)

        if candidate_value < best_value:
            best, best_value = candidate, candidate_value
            history.append(best)

    return best, best_value, history


# Simulated Annealing
def simulated_annealing(
    func, bounds, iterations=1000, temp=1000, cooling_rate=0.95, epsilon=1e-6
):
    dim = len(bounds)
    current = [random.uniform(*bounds[i]) for i in range(dim)]
    current_value = func(current)
    history = [current]

    for _ in range(iterations):
        temp *= cooling_rate
        if temp < epsilon:
            break

        neighbor = [current[i] + random.uniform(-temp, temp) for i in range(dim)]
        neighbor = [
            max(min(neighbor[i], bounds[i][1]), bounds[i][0]) for i in range(dim)
        ]
        neighbor_value = func(neighbor)

        delta = neighbor_value - current_value
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current, current_value = neighbor, neighbor_value
            history.append(current)

    return current, current_value, history


if __name__ == "__main__":
    # Межі для функції
    bounds = [(-5, 5), (-5, 5)]

    print(Fore.YELLOW + "Hill Climbing:")
    hc_solution, hc_value, hc_history = hill_climbing(sphere_function, bounds)
    print(Fore.GREEN + "Розв'язок:", hc_solution, Fore.CYAN + "Значення:", hc_value)

    print(Fore.YELLOW + "\nRandom Local Search:")
    rls_solution, rls_value, rls_history = random_local_search(sphere_function, bounds)
    print(Fore.GREEN + "Розв'язок:", rls_solution, Fore.CYAN + "Значення:", rls_value)

    print(Fore.YELLOW + "\nSimulated Annealing:")
    sa_solution, sa_value, sa_history = simulated_annealing(sphere_function, bounds)
    print(Fore.GREEN + "Розв'язок:", sa_solution, Fore.CYAN + "Значення:", sa_value)

    # Візуалізація у 3D
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Генерація сітки значень для функції сфери
    x = np.linspace(bounds[0][0], bounds[0][1], 100)
    y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    # Відображення поверхні функції
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6)

    # Функція для відображення траєкторії
    def plot_trajectory(history, color, label):
        points = np.array(history)
        ax.plot(
            points[:, 0],
            points[:, 1],
            [sphere_function(p) for p in points],
            color=color,
            marker="o",
            label=label,
        )

    # Додавання траєкторій алгоритмів
    plot_trajectory(hc_history, "red", "Hill Climbing")
    plot_trajectory(rls_history, "blue", "Random Local Search")
    plot_trajectory(sa_history, "green", "Simulated Annealing")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (Sphere Function Value)")
    ax.set_title("Пошук оптимального рішення для функції Сфери")
    ax.legend()
    plt.show()
