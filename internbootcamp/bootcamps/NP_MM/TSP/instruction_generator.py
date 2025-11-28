import random
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import os
import networkx as nx
import matplotlib.pyplot as plt

from internbootcamp.src.base_instruction_generator import BaseInstructionGenerator
from internbootcamp.bootcamps.NP_MM.prompt_md import extract_markdown_content_NP, get_prompt, get_prompt_MM

class TSPSolver:
    def __init__(self, dist_matrix, timeout=0.1):
        self.dist_matrix = dist_matrix
        self.num_cities = dist_matrix.shape[0]
        self.timeout = timeout
        self.start_time = None

    def _calculate_tour_distance(self, tour):
        return sum(self.dist_matrix[tour[i], tour[(i + 1) % self.num_cities]] for i in range(self.num_cities))

    def _initial_tour(self):
        min_dist = float('inf')
        best_tour = []
        for start_node in range(self.num_cities):
            tour = [start_node]
            unvisited = set(range(self.num_cities)) - {start_node}
            current = start_node
            while unvisited:
                nearest = min(unvisited, key=lambda n: self.dist_matrix[current, n])
                unvisited.remove(nearest)
                tour.append(nearest)
                current = nearest
            dist = self._calculate_tour_distance(tour)
            if dist < min_dist:
                min_dist, best_tour = dist, tour
        return best_tour, min_dist

    def _run_2opt(self, initial_tour):
        best_tour, best_dist = list(initial_tour), self._calculate_tour_distance(initial_tour)
        improved = True
        while improved:
            if time.time() - self.start_time > self.timeout: break
            improved, best_delta, best_swap = False, 0, None
            for i in range(self.num_cities - 1):
                for j in range(i + 2, self.num_cities):
                    u1, v1 = best_tour[i], best_tour[i+1]
                    u2, v2 = best_tour[j], best_tour[(j+1) % self.num_cities]
                    delta = (self.dist_matrix[u1, u2] + self.dist_matrix[v1, v2]) - \
                            (self.dist_matrix[u1, v1] + self.dist_matrix[u2, v2])
                    if delta < best_delta:
                        best_delta, best_swap = delta, (i, j)
            if best_swap:
                i, j = best_swap
                best_tour[i+1:j+1] = reversed(best_tour[i+1:j+1])
                best_dist += best_delta
                improved = True
        return best_tour, best_dist
    
    def solve(self):
        self.start_time = time.time()
        initial_tour, _ = self._initial_tour()
        final_tour, final_distance = self._run_2opt(initial_tour)
        return final_tour, final_distance

class NpTspInstructionGenerator(BaseInstructionGenerator):
    case_counter = 0  # 将 case_counter 定义为类变量

    def __init__(self, difficulty: Optional[str] = None, **kwargs):
        super().__init__()
        self.difficulty = kwargs.get('difficulty', difficulty)
        self.task_type = "TSP"
        self.params = kwargs
        # self.case_counter = 0 (已移除)

    def _visualize_and_save_graph(self, dist_matrix: np.ndarray, case_id: int) -> str:
        """
        Visualizes the complete graph together with its distance matrix table
        in a single, tidy figure. Nodes are arranged on a square grid for clarity.
        """
        task_name = self.task_type
        save_dir = f"data/{task_name}"
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{case_id}.png")

        G = nx.from_numpy_array(dist_matrix)
        num_nodes = G.number_of_nodes()

        # Compute square-like grid layout
        cols = int(np.ceil(np.sqrt(num_nodes)))
        rows = int(np.ceil(num_nodes / cols))
        node_list = sorted(G.nodes())
        pos = {}
        for idx, node in enumerate(node_list):
            row = idx // cols
            col = idx % cols
            nodes_in_last_row = num_nodes % cols
            if nodes_in_last_row == 0:
                nodes_in_last_row = cols
            x_offset = 0.0
            if row == rows - 1 and nodes_in_last_row < cols:
                x_offset = (cols - nodes_in_last_row) / 2.0
            x = col + x_offset
            y = rows - 1 - row
            pos[node] = (x, y)

        # Normalize positions to keep aspect square-ish
        max_x = max(p[0] for p in pos.values()) or 1
        max_y = max(p[1] for p in pos.values()) or 1
        pos = {node: (coord[0] / max_x, coord[1] / max_y) for node, coord in pos.items()}

        # Prepare figure with two panels: graph + table
        fig = plt.figure(figsize=(18, 9), facecolor='white')
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])

        ax_graph = fig.add_subplot(gs[0, 0])
        ax_graph.axis('off')
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2500,
                               edgecolors='darkblue', linewidths=3, ax=ax_graph)
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.2, alpha=0.6, ax=ax_graph)
        nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', font_color='black', ax=ax_graph)
        ax_graph.set_title("TSP Graph", fontsize=16, pad=12)
        ax_graph.set_aspect('equal')

        # Table panel
        ax_table = fig.add_subplot(gs[0, 1])
        ax_table.axis('off')
        num_cities = dist_matrix.shape[0]
        headers = [str(i) for i in range(num_cities)]
        table = ax_table.table(cellText=dist_matrix.astype(int),
                               rowLabels=headers,
                               colLabels=headers,
                               loc='center',
                               cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.1, 1.2)
        ax_table.set_title("Distance Matrix", fontsize=16, pad=12)

        plt.tight_layout()
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        return filepath

    def case_generator(self) -> Dict[str, Any]:
        NpTspInstructionGenerator.case_counter += 1
        num_cities = random.randint(*self.params["num_cities"])
        
        dist_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(i + 1, num_cities):
                dist = random.randint(*self.params["distance_range"])
                dist_matrix[i, j] = dist_matrix[j, i] = dist
        
        question = {str(i): {str(j): int(dist_matrix[i, j]) for j in range(num_cities)} for i in range(num_cities)}

        solver = TSPSolver(dist_matrix, timeout=0.1)
        _, ground_truth = solver.solve()

        image_path = self._visualize_and_save_graph(dist_matrix, NpTspInstructionGenerator.case_counter)

        return {"difficulty": self.difficulty, "question": question, "ground_truth": ground_truth, "image_path": image_path}

    def prompt_func(self, identity: Dict[str, Any]) -> str:
        md_file = "internbootcamp/bootcamps/NP_MM/TSP/TSP.md"
        task_info = extract_markdown_content_NP(md_file)
        
        q = identity["question"]
        question_str = ""
        for i in sorted(q.keys(), key=int):
            question_str += f"{i}: {q[i]}\n"
            
        prompt_txt = get_prompt_MM(self.task_type, task_info)
        prompt_txt = str(prompt_txt).strip('{}')
        prompt_img = identity["image_path"]
        prompt = {
            "prompt_txt": prompt_txt,
            "prompt_img": prompt_img,
            "question": question_str
        }
        return prompt

if __name__ == "__main__":
    generator = NpTspInstructionGenerator(
        difficulty="easy",
        num_cities=[8, 12],
        distance_range=[10, 100]
    )
    identity = generator.case_generator()
    print("Generated Identity:")
    print(identity)
    prompt = generator.prompt_func(identity)
    print("\nGenerated Prompt:")
    print(prompt)
