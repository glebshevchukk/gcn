import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
class Graph:
    def __init__(self,model):
        self.model= model
        self.graph = nx.Graph()
        self.update()

    def update(self):
        self.graph.clear()
        A = self.model.edge_weights.detach().cpu().numpy()
        self.graph = nx.from_numpy_matrix(A, create_using=nx.Graph)
        

    def draw(self):
        max_show = 50
        pos = nx.spring_layout(self.graph)
        self.graph.remove_nodes_from(range(max_show,self.model.n_hidden))
        edges,weights = zip(*nx.get_edge_attributes(self.graph,'weight').items())
        weights = list(map(abs,list(weights)))

        nx.draw_networkx_nodes(self.graph,pos,node_size=0.5)
        nx.draw_networkx_edges(self.graph,pos,edgelist=edges,edge_color=tuple(weights),width=0.5,alpha=0.7,edge_cmap=plt.cm.Blues)

def make_edge(x, y, width):
    return go.Scatter(x=x,y=y,line=dict(width=width,color='cornflowerblue'),mode = 'lines')
