from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--initial-nodes", type=int, default=6)
    parser.add_argument("--network-size", type=int, default=1000)
    parser.add_argument("--m-values", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--initial-network-type", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--replications", type=int, default=10)
    parser.add_argument("--tolerance-steps", type=int, default=31)
    parser.add_argument("--tolerance-step-size", type=float, default=0.01)
    parser.add_argument("--simulation-steps", type=int, default=500)
    parser.add_argument("--robots", type=int, default=50)
    parser.add_argument("--robot-links", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def generate_initial_network(node_count: int, initial_network_type: int) -> np.ndarray:
    if initial_network_type == 1:
        return np.zeros((node_count, node_count))
    if initial_network_type == 2:
        return np.ones((node_count, node_count))

    network = np.zeros((node_count, node_count))
    for row in range(node_count):
        for col in range(row + 1, node_count):
            if np.random.rand() > 0.5:
                network[row, col] = 1
                network[col, row] = 1
    return network


def generate_ba_network(
    initial_nodes: int,
    m_add: int,
    network_size: int,
    initial_network_type: int,
) -> np.ndarray:
    x = 100 * np.random.rand(initial_nodes)
    y = 100 * np.random.rand(initial_nodes)
    network = generate_initial_network(initial_nodes, initial_network_type)

    for new_node in range(initial_nodes, network_size):
        existing_nodes = new_node
        probabilities = np.zeros(existing_nodes)
        x = np.append(x, 100 * np.random.rand())
        y = np.append(y, 100 * np.random.rand())

        if new_node >= network.shape[0]:
            network = np.pad(network, ((0, 1), (0, 1)), mode="constant")

        for node in range(existing_nodes):
            probabilities[node] = (np.sum(network[node, :] == 1) + 1) / (
                np.sum(network == 1) + existing_nodes
            )

        cumulative_probabilities = np.cumsum(probabilities)
        visited = np.zeros(existing_nodes)
        for _ in range(m_add):
            random_value = np.random.rand()
            candidates = np.where(cumulative_probabilities >= random_value)[0]
            selected_node = candidates[0] if len(candidates) else existing_nodes - 1

            network[new_node, selected_node] = 1
            network[selected_node, new_node] = 1
            visited[selected_node] = 1

            degree = np.zeros(existing_nodes)
            total_degree = 0
            for node in range(existing_nodes):
                if visited[node] == 1:
                    probabilities[node] = 0
                    degree[node] = 0
                else:
                    degree[node] = np.sum(network[node, :] == 1) + 1
                total_degree += degree[node]

            if total_degree > 0:
                probabilities = degree / total_degree
                cumulative_probabilities = np.cumsum(probabilities)

    return network


def generate_ba_networks(args: argparse.Namespace) -> list[np.ndarray]:
    return [
        generate_ba_network(
            initial_nodes=args.initial_nodes,
            m_add=m_add,
            network_size=args.network_size,
            initial_network_type=args.initial_network_type,
        )
        for m_add in args.m_values
    ]


def initialize_network(network: np.ndarray):
    population_size = len(network)
    active_degree = np.ones(population_size)
    corpus = 1001 * np.ones((population_size, 500))
    initialized_rows = min(population_size, 1000)
    initial_comments = np.random.randint(1, 1001, (initialized_rows, 100))
    corpus[:initialized_rows, :100] = initial_comments
    opinion_climate = np.zeros(population_size)
    climate_impact = np.zeros(population_size)
    agree_sum = np.sum(corpus < 501, axis=1)
    opinion = agree_sum / np.sum(corpus <= 1000, axis=1)
    return active_degree, corpus, opinion_climate, climate_impact, opinion


def initialize_robots(network: np.ndarray, robot_count: int, robot_links: int):
    population_size = len(network)
    robot_links = min(robot_links, population_size)
    robots = np.zeros((robot_links, robot_count), dtype=int)
    degree = np.sum(network, axis=1)
    ordered_index = np.argsort(degree)[::-1]
    connected_nodes = ordered_index[:robot_links]

    for robot_index in range(robot_count):
        robots[:, robot_index] = connected_nodes

    robot_corpus = 1001 * np.ones((robot_count, 500))
    robot_corpus[:robot_count, :100] = np.tile(np.arange(1, 101), (robot_count, 1))
    agree_sum = np.sum(robot_corpus < 501, axis=1)
    robot_opinion = agree_sum / np.sum(robot_corpus <= 1000, axis=1)
    robot_activity = 50 * np.ones(robot_count)
    robot_impact = np.zeros(robot_count)
    return robots, robot_corpus, robot_opinion, robot_activity, robot_impact


def activate_robots(robots: np.ndarray, robot_opinion: np.ndarray, opinion: np.ndarray) -> np.ndarray:
    robot_count = robots.shape[1]
    robot_impact = np.zeros(robot_count)
    active_robots = np.zeros(robot_count)

    for robot_index in range(robot_count):
        connected_nodes = robots[:, robot_index]
        opinion_gap = np.sum(robot_opinion[robot_index] - opinion[connected_nodes]) / len(connected_nodes)
        robot_impact[robot_index] = opinion_gap

    for robot_index in range(robot_count):
        sigmoid = 1 / (1 + np.exp(-robot_impact[robot_index]))
        active_robots[robot_index] = 1 if np.random.rand() < sigmoid else 0

    return np.where(active_robots != 0)[0]


def add_active_robots_to_network(
    carrier_network: np.ndarray,
    active_robot_order: np.ndarray,
    robots: np.ndarray,
    robot_corpus: np.ndarray,
    robot_activity: np.ndarray,
    robot_opinion: np.ndarray,
    corpus: np.ndarray,
    active_degree: np.ndarray,
    opinion: np.ndarray,
):
    robot_count = len(active_robot_order)
    if robot_count == 0:
        return carrier_network, corpus, active_degree, opinion

    current_size = carrier_network.shape[0]
    merge_matrix = np.zeros((current_size, robot_count))
    for index, robot_index in enumerate(active_robot_order):
        merge_matrix[robots[:, robot_index], index] = 1

    carrier_network = np.column_stack((carrier_network, merge_matrix))
    symmetric_rows = np.column_stack((merge_matrix.T, np.zeros((robot_count, robot_count))))
    carrier_network = np.vstack((carrier_network, symmetric_rows))

    for robot_index in active_robot_order:
        corpus = np.vstack((corpus, robot_corpus[robot_index, :]))
        active_degree = np.hstack((active_degree, robot_activity[robot_index]))
        opinion = np.hstack((opinion, robot_opinion[robot_index]))

    return carrier_network, corpus, active_degree, opinion


def update_opinion_climate(
    carrier_network: np.ndarray,
    population_size: int,
    opinion: np.ndarray,
    opinion_climate: np.ndarray,
    climate_impact: np.ndarray,
) -> None:
    for node in range(population_size):
        connected_nodes = np.where(carrier_network[:, node] != 0)[0]
        if len(connected_nodes) == 0:
            opinion_climate[node] = 0
        else:
            neighbors_opinion = opinion[connected_nodes]
            opinion_climate[node] = abs(np.sum(opinion[node] - neighbors_opinion) / len(connected_nodes))
        climate_impact[node] = (2 / (1 + np.exp(-2 * opinion_climate[node]))) - 1


def choose_active_nodes(climate_impact: np.ndarray, population_size: int) -> np.ndarray:
    activation_threshold = np.random.rand(population_size)
    return np.where(climate_impact - activation_threshold > 0)[0]


def choose_communication_nodes(
    carrier_network: np.ndarray,
    active_nodes: np.ndarray,
    active_degree: np.ndarray,
) -> list[int]:
    communication_nodes: list[int] = []
    active_neighbors = carrier_network[:, active_nodes]

    for index in range(len(active_nodes)):
        neighbors = np.where(active_neighbors[:, index] != 0)[0]
        if len(neighbors) == 0:
            communication_nodes.append(active_nodes[index])
            continue

        neighbor_activity = active_degree[neighbors]
        cumulative_activity = np.cumsum(neighbor_activity)
        random_position = np.random.rand() * np.max(cumulative_activity)
        selected_index = np.argmin(np.abs(cumulative_activity - random_position))
        selected_value = cumulative_activity[selected_index]
        selected_position = np.where(cumulative_activity == selected_value)[0]
        communication_nodes.extend(neighbors[selected_position])

    return communication_nodes


def interact_opinions(
    active_nodes: np.ndarray,
    communication_nodes: list[int],
    active_degree: np.ndarray,
    corpus: np.ndarray,
) -> None:
    previous_corpus = corpus.copy()
    for index, active_node in enumerate(active_nodes):
        communication_node = communication_nodes[index]
        communication_count = int(np.floor(active_degree[communication_node]))
        if communication_count <= 0:
            continue

        language = previous_corpus[communication_node, :]
        language = language[language != 1001]
        origin_language = previous_corpus[active_node, :]
        origin_language = origin_language[origin_language != 1001]

        if len(language) == 0:
            continue
        communication_count = min(communication_count, len(language))

        if len(origin_language) + communication_count > 500:
            dropout_count = len(origin_language) + communication_count - 500
            origin_language = origin_language[dropout_count:]

        selected_language = np.random.choice(language, communication_count, replace=False)
        selected_language = np.sort(selected_language)
        origin_language = np.concatenate((origin_language, selected_language))

        if len(origin_language) < 500:
            origin_language = np.concatenate((origin_language, 1001 * np.ones(500 - len(origin_language))))

        corpus[active_node, :] = origin_language


def simulate_once(
    base_network: np.ndarray,
    tolerance: float,
    simulation_steps: int,
    robot_count: int,
    robot_links: int,
) -> float:
    carrier_network = base_network.copy()
    population_size = len(carrier_network)
    active_degree, corpus, opinion_climate, climate_impact, opinion = initialize_network(carrier_network)
    np.fill_diagonal(carrier_network, 0)
    robots, robot_corpus, robot_opinion, robot_activity, _ = initialize_robots(
        carrier_network,
        robot_count=robot_count,
        robot_links=robot_links,
    )

    for _ in range(simulation_steps):
        active_robot_order = activate_robots(robots, robot_opinion, opinion)
        carrier_network, corpus, active_degree, opinion = add_active_robots_to_network(
            carrier_network,
            active_robot_order,
            robots,
            robot_corpus,
            robot_activity,
            robot_opinion,
            corpus,
            active_degree,
            opinion,
        )
        update_opinion_climate(carrier_network, population_size, opinion, opinion_climate, climate_impact)
        active_nodes = choose_active_nodes(climate_impact, population_size)
        communication_nodes = choose_communication_nodes(carrier_network, active_nodes, active_degree)
        interact_opinions(active_nodes, communication_nodes, active_degree, corpus)
        active_degree = 0.8 * active_degree

    agree_sum = np.sum(corpus[:population_size, :] < 501, axis=1)
    final_opinion = agree_sum / np.sum(corpus[:population_size, :] <= 1000, axis=1)
    return float(np.mean(final_opinion))


def run_experiment(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    networks = generate_ba_networks(args)
    tolerance_values = np.arange(args.tolerance_steps) * args.tolerance_step_size
    all_results = np.zeros((args.replications, len(tolerance_values), len(args.m_values)))

    for network_index, network in enumerate(networks):
        for replication in range(args.replications):
            for tolerance_index, tolerance in enumerate(tolerance_values):
                all_results[replication, tolerance_index, network_index] = simulate_once(
                    base_network=network,
                    tolerance=tolerance,
                    simulation_steps=args.simulation_steps,
                    robot_count=args.robots,
                    robot_links=args.robot_links,
                )

    return all_results.mean(axis=0), tolerance_values


def save_results_csv(result_matrix: np.ndarray, tolerance_values: np.ndarray, m_values: list[int], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["tolerance", *[f"m={m_value}" for m_value in m_values]])
        for row_index, tolerance in enumerate(tolerance_values):
            writer.writerow([f"{tolerance:.4f}", *[f"{value:.8f}" for value in result_matrix[row_index]]])


def plot_heatmap(
    result_matrix: np.ndarray,
    tolerance_values: np.ndarray,
    m_values: list[int],
    out_path: Path,
    show: bool,
) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        result_matrix,
        xticklabels=m_values,
        yticklabels=[f"{value:.2f}" for value in tolerance_values],
        cmap="hot",
        vmin=0.5,
        vmax=1,
        cbar_kws={"label": "Value"},
        annot=True,
        fmt=".4f",
        annot_kws={"size": 8},
        cbar=True,
    )
    plt.xlabel("m-parameters of BA scale-free networks ($m$)")
    plt.ylabel("Social bot tolerance ($\\theta$)")
    plt.title("Heatmap of Simulation Results")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_matrix, tolerance_values = run_experiment(args)
    save_results_csv(result_matrix, tolerance_values, args.m_values, args.output_dir / "simulation_results.csv")
    plot_heatmap(result_matrix, tolerance_values, args.m_values, args.output_dir / "simulation_heatmap.png", args.show)

    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
