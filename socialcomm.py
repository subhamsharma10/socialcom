import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import numpy as np

# 1. Load Dataset - Zachary's Karate Club Graph
G = nx.karate_club_graph()
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# 2. Network Properties
degrees = [d for n, d in G.degree()]
print(f"Average Degree  : {sum(degrees)/len(degrees):.2f}")
print(f"Max Degree      : {max(degrees)}")
print(f"Density         : {nx.density(G):.4f}")
print(f"Avg Clustering  : {nx.average_clustering(G):.4f}")
print(f"Is Connected    : {nx.is_connected(G)}")

# 3. Apply Louvain Community Detection
partition = community_louvain.best_partition(G, random_state=42)
num_communities = len(set(partition.values()))
print(f"Communities Detected: {num_communities}")

# 4. Modularity Score
modularity = community_louvain.modularity(partition, G)
print(f"Modularity Score: {modularity:.4f}")

# 5. Community Breakdown
sizes = Counter(partition.values())
for c_id, size in sorted(sizes.items()):
    members = [n for n, c in partition.items() if c == c_id]
    print(f"Community {c_id+1}: {size} nodes -> {members}")

# 6. Visualize Communities
COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
color_map = [COLORS[partition[n]] for n in G.nodes()]
pos = nx.spring_layout(G, seed=42, k=0.6)

plt.figure(figsize=(10, 8))
nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.8, edge_color='grey')
nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=280, alpha=0.9)
nx.draw_networkx_labels(G, pos, font_size=7, font_color='white', font_weight='bold')
patches = [mpatches.Patch(color=COLORS[i],
           label=f'Community {i+1} ({sizes[i]} nodes)')
           for i in range(num_communities)]
plt.legend(handles=patches, loc='upper left', fontsize=9)
plt.title('Social Network - Community Detection (Louvain)')
plt.axis('off')
plt.savefig('community_graph.png', dpi=150, bbox_inches='tight')
plt.show()

# 7. Degree Distribution & Community Sizes
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(degrees, bins=12, color='#3498DB', edgecolor='white')
axes[0].set_title('Degree Distribution')
axes[0].set_xlabel('Degree')
axes[0].set_ylabel('Frequency')

comm_sizes = [sizes[c] for c in sorted(sizes.keys())]
bars = axes[1].bar([f'C{i+1}' for i in range(num_communities)],
                    comm_sizes, color=COLORS, edgecolor='white')
axes[1].set_title('Nodes per Community')
axes[1].set_xlabel('Community')
axes[1].set_ylabel('Number of Nodes')
plt.tight_layout()
plt.savefig('network_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# 8. Adjacency Matrix (grouped by community)
nodes_sorted = sorted(G.nodes(), key=lambda n: partition[n])
A = nx.to_numpy_array(G, nodelist=nodes_sorted)
plt.figure(figsize=(8, 7))
plt.imshow(A, cmap='Blues', aspect='auto')
plt.title('Adjacency Matrix (nodes grouped by community)')
plt.colorbar()
plt.savefig('adjacency_matrix.png', dpi=150, bbox_inches='tight')
plt.show()