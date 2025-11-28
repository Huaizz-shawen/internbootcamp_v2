from typing import Dict, Any, Optional, List, Tuple
from internbootcamp.src.base_instruction_generator import BaseInstructionGenerator
from internbootcamp.bootcamps.NP_MM.prompt_md import extract_markdown_content_NP, get_prompt_MM
import random
import time
from collections import defaultdict
import os
import networkx as nx
import matplotlib.pyplot as plt

class GCPSolver:
    """
    A solver for the Graph Coloring Problem (GCP-D) using a powerful hybrid
    heuristic approach:
    1. DSatur algorithm for a high-quality initial coloring.
    2. Tabu Search to iteratively try and reduce the number of colors used.
    """
    def __init__(self, adj_list, timeout=60):
        self.adj_list = {int(k): v for k, v in adj_list.items()}
        self.nodes = sorted(self.adj_list.keys())
        self.num_vertices = len(self.nodes)
        self.timeout = timeout
        self.start_time = None

    def _dsatur_initial_coloring(self):
        if not self.nodes:
            return {}, 0
        saturation = {node: 0 for node in self.nodes}
        degrees = {node: len(self.adj_list.get(node, [])) for node in self.nodes}
        coloring = {}
        uncolored_nodes = set(self.nodes)
        while uncolored_nodes:
            node_to_color = max(uncolored_nodes, key=lambda u: (saturation[u], degrees[u]))
            uncolored_nodes.remove(node_to_color)
            neighbor_colors = {coloring[n] for n in self.adj_list.get(node_to_color, []) if n in coloring}
            color = 1
            while color in neighbor_colors:
                color += 1
            coloring[node_to_color] = color
            for neighbor in self.adj_list.get(node_to_color, []):
                if neighbor in uncolored_nodes:
                    neighbor_neighbor_colors = {coloring[n] for n in self.adj_list.get(neighbor, []) if n in coloring}
                    saturation[neighbor] = len(neighbor_neighbor_colors)
        num_colors = max(coloring.values()) if coloring else 0
        return coloring, num_colors

    def _tabu_search_for_k_colors(self, k, base_coloring, max_iter=10000):
        current_coloring = dict(base_coloring)
        colors_to_nodes = defaultdict(list)
        for node, color in current_coloring.items():
            colors_to_nodes[color].append(node)
        sorted_color_classes = sorted(colors_to_nodes.items(), key=lambda item: len(item[1]))
        if len(sorted_color_classes) > 1:
            color_to_remove = sorted_color_classes[0][0]
            nodes_to_recolor = sorted_color_classes[0][1]
            target_color = sorted_color_classes[1][0]
            for node in nodes_to_recolor:
                current_coloring[node] = target_color
        distinct_colors = sorted(list(set(current_coloring.values())))
        color_map = {old_color: new_color for new_color, old_color in enumerate(distinct_colors, 1)}
        current_coloring = {node: color_map[color] for node, color in current_coloring.items()}
        conflicts = set()
        for u in self.nodes:
            for v in self.adj_list.get(u, []):
                if u < v and current_coloring[u] == current_coloring[v]:
                    conflicts.add(tuple(sorted((u, v))))
        tabu_list = {}
        tabu_tenure = min(15, self.num_vertices // 4 + 1)
        for i in range(max_iter):
            if not conflicts:
                return current_coloring
            if time.time() - self.start_time > self.timeout:
                raise TimeoutError
            best_move = None
            best_delta = float('inf')
            conflicting_nodes = list(set(u for u, v in conflicts).union(v for u, v in conflicts))
            random.shuffle(conflicting_nodes)
            for node in conflicting_nodes:
                current_color = current_coloring[node]
                for new_color in range(1, k + 1):
                    if new_color == current_color:
                        continue
                    if (node, new_color) in tabu_list and tabu_list[(node, new_color)] > i:
                        continue
                    old_conflicts = sum(1 for neighbor in self.adj_list.get(node, []) if current_coloring[neighbor] == current_color)
                    new_conflicts = sum(1 for neighbor in self.adj_list.get(node, []) if current_coloring[neighbor] == new_color)
                    delta = new_conflicts - old_conflicts
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (node, new_color)
            if best_move:
                node_to_move, new_color = best_move
                old_color = current_coloring[node_to_move]
                for neighbor in self.adj_list.get(node_to_move, []):
                    edge = tuple(sorted((node_to_move, neighbor)))
                    if current_coloring[neighbor] == old_color: conflicts.remove(edge)
                    if current_coloring[neighbor] == new_color: conflicts.add(edge)
                current_coloring[node_to_move] = new_color
                tabu_list[(node_to_move, old_color)] = i + tabu_tenure
        return None

    def solve(self):
        self.start_time = time.time()
        best_coloring, k = self._dsatur_initial_coloring()
        if not best_coloring:
            return None, 0
        try:
            for target_k in range(k - 1, 0, -1):
                if time.time() - self.start_time > self.timeout:
                    break
                new_solution = self._tabu_search_for_k_colors(target_k, best_coloring)
                if new_solution:
                    best_coloring = new_solution
                    k = target_k
                else:
                    break
        except TimeoutError:
            pass
        final_coloring_list = [0] * self.num_vertices
        for node, color in best_coloring.items():
            final_coloring_list[node] = color
        return final_coloring_list, k

class NpGcpDInstructionGenerator(BaseInstructionGenerator):
    case_counter = 0  # 将 case_counter 定义为类变量

    def __init__(self, ground_truth: Optional[float] = None, difficulty: Optional[str] = None, **kwargs):
        super().__init__()
        self.ground_truth = ground_truth
        self.difficulty = kwargs.get('difficulty', difficulty)
        self.task_type = "gcp-d"
        self.params = kwargs
        # self.case_counter = 0 (已移除)

    def _generate_single_question(self) -> Tuple[Dict, List[int]]:
        num_vertices = random.randint(*self.params["num_vertices"])
        target_chromatic_number = random.randint(*self.params["num_colors"])

        # 1. Create k partitions for a k-colorable graph
        partitions = [[] for _ in range(target_chromatic_number)]
        coloring_map = {}
        nodes = list(range(num_vertices))
        random.shuffle(nodes)
        for i, node in enumerate(nodes):
            color_index = i % target_chromatic_number
            partitions[color_index].append(node)
            coloring_map[node] = color_index + 1

        adj = {str(i): [] for i in range(num_vertices)}
        
        # 2. Add edges *between* partitions to ensure k-colorability
        # This creates the base structure of the coloring problem.
        edge_density = self.params["edge_density"]
        for i in range(target_chromatic_number):
            for j in range(i + 1, target_chromatic_number):
                for u in partitions[i]:
                    for v in partitions[j]:
                        if random.random() < edge_density:
                            adj[str(u)].append(v)
                            adj[str(v)].append(u)

        # 3. For harder difficulties, add "confusing" edges inside partitions
        # This forces the chromatic number to increase and creates challenges.
        if self.difficulty in ["hard", "bench"]:
            num_confusing_edges = int(num_vertices * self.params.get("confusion_factor", 0.1))
            for _ in range(num_confusing_edges):
                # Pick a random partition and add an edge within it
                partition_to_disrupt = random.choice(partitions)
                if len(partition_to_disrupt) >= 2:
                    u, v = random.sample(partition_to_disrupt, 2)
                    # Check if edge already exists to avoid duplicates
                    if v not in adj[str(u)]:
                        adj[str(u)].append(v)
                        adj[str(v)].append(u)

        final_adj = {str(k): sorted(list(set(v))) for k, v in adj.items()}
        
        # Note: The original 'coloring' is now just a starting point.
        # The true ground truth needs to be found by the solver, especially after adding confusing edges.
        return final_adj, list(coloring_map.values())

    def _visualize_and_save_graph(self, graph: Dict[str, List[int]], case_id: int) -> str:
        """
        Visualizes the graph using networkx and matplotlib with a grid layout.
        Creates a clean, organized grid-based layout.
        """
        task_name = self.task_type
        save_dir = f"data/{task_name}"
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{case_id}.png")

        G = nx.Graph()
        for node, neighbors in graph.items():
            G.add_node(int(node))
            for neighbor in neighbors:
                G.add_edge(int(node), int(neighbor))

        num_nodes = len(G.nodes())
        
        # Calculate grid dimensions (rows x cols)
        cols = 4
        rows = (num_nodes + cols - 1) // cols  # Ceiling division
        
        # Create grid positions
        pos = {}
        node_list = sorted(G.nodes())
        
        for idx, node in enumerate(node_list):
            row = idx // cols
            col = idx % cols
            
            nodes_in_last_row = num_nodes % cols
            if nodes_in_last_row == 0:
                nodes_in_last_row = cols
            
            # Center the last row if it has fewer nodes
            if row == rows - 1 and nodes_in_last_row < cols:
                offset = (cols - nodes_in_last_row) / 2.0
                x = (col + offset) * 2.0
            else:
                x = col * 2.0
            
            y = (rows - 1 - row) * 2.0  # Invert y for top-down layout
            pos[node] = (x, y)

        # Create figure
        fig_size = max(12, cols * 3), max(12, rows * 3)
        plt.figure(figsize=fig_size, facecolor='white')
        
        # Draw with larger, clearer styling
        nx.draw_networkx_nodes(
            G, pos,
            node_color='lightblue',
            node_size=3000,
            edgecolors='darkblue',
            linewidths=3
        )
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            width=2.5,
            alpha=0.7
        )
        nx.draw_networkx_labels(
            G, pos,
            font_size=16,
            font_weight='bold'
        )
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filepath

    def case_generator(self) -> Dict[str, Any]:
        identity = {"difficulty": self.difficulty}
        
        NpGcpDInstructionGenerator.case_counter += 1
        graph, _ = self._generate_single_question()
        image_path = self._visualize_and_save_graph(graph, NpGcpDInstructionGenerator.case_counter)

        solver = GCPSolver(graph, timeout=1) # Using a 1s timeout for generation
        _, ground_truth = solver.solve()
        
        identity["question"] = graph
        identity["ground_truth"] = ground_truth
        identity["image_path"] = image_path
        return identity

    def prompt_func(self, identity: Dict[str, Any]) -> str:
        md_file = "internbootcamp/bootcamps/NP_MM/GCP_D/GCP-D.md"
        task_info = extract_markdown_content_NP(md_file)
        
        # Format the graph for the prompt
        graph_str = ""
        sorted_nodes = sorted(identity["question"].keys(), key=int)
        for node in sorted_nodes:
            neighbors = identity["question"][node]
            graph_str += f"{node}: {neighbors}\n"
            
        prompt_txt = get_prompt_MM(self.task_type, task_info)
        prompt_txt = str(prompt_txt).strip('{}')
        prompt_img = identity["image_path"]
        prompt = {
            "prompt_txt": prompt_txt,
            "prompt_img": prompt_img,
            "question": graph_str
        }
        return prompt

if __name__ == "__main__":
    generator = NpGcpDInstructionGenerator(difficulty="hard") # Test with 'hard'
    num_vertices = [15, 20]
    num_colors = [3, 4]
    edge_density = 0.3
    confusion_factor = 0.15 # Add confusion factor for hard/bench
    generator.params["num_vertices"] = num_vertices
    generator.params["num_colors"] = num_colors
    generator.params["edge_density"] = edge_density
    generator.params["confusion_factor"] = confusion_factor
    identity = generator.case_generator()
    print("Generated Identity:")
    print(identity)
    prompt = generator.prompt_func(identity)
    print("\nGenerated Prompt:")
    print(prompt)
