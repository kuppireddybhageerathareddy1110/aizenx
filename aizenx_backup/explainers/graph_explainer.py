import numpy as np
import networkx as nx


class GraphExplainer:

    def __init__(self, interactions, feature_names):

        self.interactions = interactions
        self.feature_names = feature_names


    def build_graph(self, threshold=0.01):

        G = nx.Graph()

        for name in self.feature_names:
            G.add_node(name)

        n = len(self.feature_names)

        for i in range(n):
            for j in range(i+1, n):

                weight = abs(self.interactions[i, j])

                if weight > threshold:

                    G.add_edge(
                        self.feature_names[i],
                        self.feature_names[j],
                        weight=weight
                    )

        return G