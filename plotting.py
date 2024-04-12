import matplotlib.pyplot as plt
import networkx as nx

class Plotting:

    def scatter_plot_principal_components(principal_components):
        plt.figure(figsize=(12, 8))
        plt.scatter(principal_components[:, 0], principal_components[:, 1], s=50, alpha=0.5)
        plt.title('Scatter Plot of Principal Components')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        plt.show()
        
    def scatter_plot_with_clusters(principal_components, clusters):
        plt.figure(figsize=(12, 8))
        for cluster in range(max(clusters) + 1):
            subset = principal_components[clusters == cluster]
            plt.scatter(subset[:, 0], subset[:, 1], s=50, label=f'Cluster {cluster}', alpha=0.5)
        plt.title('Scatter Plot of Principal Components with k-Means Clusters')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_cluster_graph(principal_components, clusters):
        G = nx.Graph()

        for i, pc in enumerate(principal_components):
            G.add_node(i, pos=(pc[0], pc[1]), cluster=clusters[i])

        for i in range(len(principal_components)):
            for j in range(i + 1, len(principal_components)):
                if clusters[i] == clusters[j]:
                    G.add_edge(i, j)

        pos = nx.get_node_attributes(G, 'pos')
        cluster_colors = {0: 'r', 1: 'g', 2: 'b', 3: 'y', 4: 'c', 5: 'm'}

        plt.figure(figsize=(12, 8))
        for cluster in range(max(clusters) + 1):
            nodes = [node for node, attrs in G.nodes(data=True) if attrs['cluster'] == cluster]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=cluster_colors[cluster], node_size=100, alpha=0.7, label=f'Cluster {cluster}')
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        plt.title('Graph of Clusters with GCN')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)
        plt.show()
