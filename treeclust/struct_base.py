import numpy as np
import igraph
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .utils import *
import tqdm

class RobustClustering():

    parameter_range = None
    random_state = None
    n_iter = None
    threshold = 0.5
    additional_metrics = {}
    matrix = None
    connectivity_matrix = None
    marker_genes_matrix = None
    annotation_dict = None    
    clustering_object = None
    stop_criterion = None
    stop_threshold = None
    probability_metric = None
    _cluster_resolutions = []
    _fuzzy_graph = None
    _active_clusters = np.array([0])
    _counts = [1]
    _metrics = {"stability":None}
    
    def __init__(
        self, 
        parameter_range=np.arange(0,10,0.1), 
        random_state=0, 
        n_iter=10, 
        threshold=0.5, 
        additional_metrics={},
        X=None,
        connectivity_matrix=None,
        marker_genes_matrix=None,
        annotation_dict=None,
    ):

        self.parameter_range = list(parameter_range)
        self.random_state = random_state
        self.n_iter = n_iter
        self.threshold = threshold
        self.additional_metrics = additional_metrics
        self.X = X
        self.connectivity_matrix = connectivity_matrix
        self.marker_genes_matrix = marker_genes_matrix
        self.annotation_dict = annotation_dict
        
    def plot_graph(
        self, 
        y=None, 
        color=None, 
        size=None, 
        vertex_label="name", 
        size_norm=(10,10), 
        vertex_label_size=10, 
        palette="rocket", 
        invert=False,
        **kwargs        
    ):

        g = self._fuzzy_graph
    
        if color != None:
            m = {i:sns.color_palette(palette,11)[i] for i in range(11)}
            m_ = max(g.vs()[color])
            c = [m[int(i/m_*10)] for i in g.vs()[color]]
        else:
            c = color

        if size != None:
            min_ = np.max(g.vs()[size])
            max_ = np.min(g.vs()[size])
            s = [(i-min_)/(max_-min_)*(size_norm[1]-size_norm[0])+size_norm[0] for i in g.vs()[size]]      
        else:
            s = size

        if vertex_label != None:
            name = g.vs()[vertex_label]
        else: 
            name = vertex_label

        shape = np.array(["circle"]*len(self._cluster_resolutions), "U7")
        shape[self._active_clusters] = "diamond"
        
        visual_style = {
            "vertex_shape":shape,
            "vertex_color":c,
            "vertex_size":s,
            "vertex_label":name,
            "vertex_label_size":vertex_label_size,
            "edge_arrow_size":0,
        }
        for i,j in kwargs.items():
            visual_style[i] = j
    
        layout = g.layout_reingold_tilford(root=[0], mode="out")
        if y != None:
            s = 1
            if invert:
                s = -1
            layout = [[c[0], s*g.vs()[y][i]] for i,c in enumerate(layout.coords)]

        f = igraph.plot(self._fuzzy_graph, layout=layout, **visual_style)
            
        return f
        
    def plot_annotation_markers(self, metric=None, figsize=None):

        if metric == None:
            metric = self.probability_metric
        
        marker_genes_matrix = self.marker_genes_matrix
        annotation_dict = self.annotation_dict
        n_cells, n_clusters = self._metrics[metric].shape
        s = [len(i["genes"]) for _,i in annotation_dict.items()]
        n_columns = sum(s)
    
        m = marker_genes_matrix.values.transpose().dot(self._metrics[metric])/self._metrics[metric].sum(axis=0)
        m = pd.DataFrame(m.transpose(),columns=marker_genes_matrix.columns.values)
        m = m.div(m.max(axis=0),axis=1)
    
        if figsize == None:
            figsize = (10*n_columns/n_clusters,5)
            print(figsize)
        
        fig, ax = plt.subplots(1,len(annotation_dict),figsize=figsize, width_ratios=s)
        for c, (i,j) in enumerate(annotation_dict.items()):
            l = j["genes"]
    
            sns.heatmap(m.loc[:,l], cbar=False, ax=ax[c])
            ax[c].set_xlabel(i)
        
        return fig, ax
    
    def set_marker_genes(self, marker_genes_matrix):

        self.marker_genes_matrix = marker_genes_matrix

    def set_annotation_dict(self, annotation_dict):

        self.annotation_dict = annotation_dict

    def score_annotation(self, annotation_dict=None, marker_genes_matrix=None, metric=None, vote_style = "continuous", threshold = 0):

        if metric == None:
            metric = self.probability_metric
        
        if type(marker_genes_matrix)==type(None):
            marker_genes_matrix = self.marker_genes_matrix
        gene_names = marker_genes_matrix.columns.values
        
        if type(annotation_dict)==type(None):
            annotation_dict = self.annotation_dict
        
        if vote_style == "discrete":
            votes = marker_genes_matrix.values>threshold
        elif vote_style == "continuous":
            votes = np.divide(marker_genes_matrix.values,marker_genes_matrix.values.max(axis=0))
        else:
            raise ValueError(f"'vote_style' should be 'discrete' or 'continuous'.")
        
        n_cells, n_clusters = self._metrics[metric].shape
        p = self._metrics[metric]/self._metrics[metric].sum(axis=0)
        size = p.sum(axis=0)
        specificity_global = ((votes.transpose().dot(self._metrics[metric])).transpose()/votes.sum(axis=0)).transpose()
        specificity_local = (votes.transpose().dot(self._metrics[metric]))/self._metrics[metric].sum(axis=0)
        specificity_global = pd.DataFrame(specificity_global, index = gene_names)
        specificity_local = pd.DataFrame(specificity_local, index = gene_names)
        sgl = specificity_global*specificity_local
        
        n_categories = len(annotation_dict)
        d = pd.DataFrame(np.zeros([n_clusters, n_categories]), columns=[i for i in annotation_dict.keys()])
        for i,j in annotation_dict.items():
            j = j["genes"]
            for _,k in enumerate(j):
                for l in range(n_clusters):
                    c = sgl.loc[annotation_dict[i]["genes"],l]
                    d.loc[l,i] += c.loc[k]/len(j)
            
        return d
            
    def split(self, cluster):

        if cluster in self._active_clusters:
            daughters = [i.target for i in self._fuzzy_graph.es.select(_source=cluster)]
            if len(daughters) > 0:
                vs = self._fuzzy_graph.vs(daughters)
                m = [i["name"] for i in vs]
                self._active_clusters = np.concatenate([self._active_clusters,m])
                self._active_clusters = self._active_clusters[self._active_clusters != cluster]
            else:
                raise ValueError(f"Cluster {cluster} has not daughter nodes so it cannot be split.")
        else:
            raise ValueError(f"Cluster {cluster} cannot be split because is not inside the active clusters.")

    def merge(self, cluster):

        if cluster in self._active_clusters:
            parent = [i.source for i in self._fuzzy_graph.es.select(_target=cluster)]
            if len(parent)>0:
                parent = parent[0]
                daughters = [i.target for i in self._fuzzy_graph.es.select(_source=parent)]
                
                self._active_clusters = np.concatenate([self._active_clusters,[parent]])
                self._active_clusters = self._active_clusters[[i not in daughters for i in self._active_clusters]]
            else:
                raise ValueError(f"Cluster {cluster} has not parent nodes so it cannot be merged.")
        else:
            raise ValueError(f"Cluster {cluster} cannot be split because is not inside the active clusters.")

    def set_threshold(self, metric=None, threshold=0.5, maximum_likelihood=True):

        if metric == None:
            metric = self.probability_metric
        
        m = np.array(self._fuzzy_graph.vs()[metric]) > threshold
        keep = np.where(m)[0]
        gs = self._fuzzy_graph.subgraph(keep)
        for g in gs.connected_components(mode="weak"):
            if 0 in gs.vs()["name"]:
                break
    
        sg = self._fuzzy_graph.subgraph(gs.subgraph(g).vs()["name"])
        active_clusters = [i.index for i in sg.vs.select(_outdegree = 0)]
        if maximum_likelihood:
            pruning = True
            while pruning:
                pruning = False
                active_clusters_ = []
                for i in active_clusters:
                    source = sg.es.select(_target=i)[0].source
                    if sg.vs(source).outdegree()[0] == 1:
                        pruning = True
                        active_clusters_.append(source)
                    else:
                        active_clusters_.append(i)
                active_clusters = active_clusters_.copy()
    
        self._active_clusters = np.array(sg.vs(active_clusters)["name"])
    
    def get_active_clusters(self):

        return self._active_clusters.copy()
    
    def predict(self, metric = None):

        if metric == None:
            metric = self.probability_metric
        
        probabilities = self._metrics[metric][:,self._active_clusters]
        probabilities = (probabilities.transpose() / probabilities.sum(axis=1)).transpose()

        return self._active_clusters[probabilities.argmax(axis=1)]

    def predict_probability(self, metric = None):

        if metric == None:
            metric = self.probability_metric
        
        probabilities = self._metrics[metric][:,self._active_clusters]
        probabilities = (probabilities.transpose() / probabilities.sum(axis=1)).transpose()

        return probabilities

    def set_annotation(self, annotation):

        self.annotation = annotation
    
    def get_metric(self, metric):

        return self._fuzzy_graph.vs()[metric]
    
    def _fit_parameter(self, res):

        counts = list(self._counts)
        cluster_resolutions = list(self._cluster_resolutions)
        for count,i in enumerate(range(self.n_iter)):
            
            v = self._find_clustering(res, self.random_state)
            
            membership = get_membership_(v)

            votes = self._metrics["stability"].shape[1]-1-np.argmax((membership.transpose().dot(self._metrics["stability"])/membership.sum(axis=0).reshape(-1,1)>0.5)[:,::-1],axis=1)
            
            for j in np.unique(votes):
                
                divisions = j == votes
                if sum(divisions) > 1:
                    
                    self._metrics["stability"] = np.concatenate([self._metrics["stability"], membership[:,divisions]],axis=1)
                    
                    # Compute other metrics
                    for metric_name, metric in self.additional_metrics.items():
                        p = metric(membership, self)
                        try:
                            for div in np.where(divisions)[0]:
                                self._metrics[metric_name] = np.concatenate([self._metrics[metric_name], p[:,div].reshape(-1,1)],axis=1)
                        except:
                            raise Exception(f"Metric {metric_name} not computed. Missing arguments.")
                            
                    self._fuzzy_graph.add_vertices(sum(divisions))        
                    self._fuzzy_graph.add_edges([(j,k) for k in range(len(counts),len(counts)+sum(divisions))])
                    counts += [1]*sum(divisions)
                    cluster_resolutions += [res]*sum(divisions)
                    
                else:
                    
                    if max(self._cluster_resolutions) == cluster_resolutions[j]:
                        k = np.where(divisions)[0][0]
                        
                        self._metrics["stability"][:,j] = self._metrics["stability"][:,j]*counts[j]/(counts[j]+1) + membership[:,k]/(counts[j]+1)
                        
                        for metric_name, metric in self.additional_metrics.items():
                            p = metric(membership, self)
                            try:
                                self._metrics[metric_name][:,j] = self._metrics[metric_name][:,j]*counts[j]/(counts[j]+1) + p[:,k]/(counts[j]+1)
                            except:
                                raise Exception(f"Metric {metric_name} not computed. Missing arguments.")

                        counts[j] += 1

        self._counts = np.array(counts)
        self._cluster_resolutions = np.array(cluster_resolutions)
    
    def fit(self, verbose = True):

        self._metrics["stability"] = np.ones([self.connectivity_matrix.shape[0],1],np.float64)
        
        for metric_name, metric in self.additional_metrics.items():
            try:
                self._metrics[metric_name] = metric(self._metrics["stability"], self)
            except:
                 raise Exception(f"Metric {metric_name} not computed. Missing arguments.")
                
        self._counts = [1]
        self._cluster_resolutions = [-0.001]
        self._fuzzy_graph = igraph.Graph(n=1,directed=True)
        
        if verbose:
            iter = tqdm.tqdm(self.parameter_range)
        else:
            iter = self.parameter_range
            
        for resolution in iter:
            self._fit_parameter(resolution)

            p = self._metrics[self.stop_criterion].copy()
            p[p==0] = np.nan
            p = np.nanmean(p,axis=0)
            if np.all(p[self._cluster_resolutions == self._cluster_resolutions[-1]] < self.stop_threshold):
                break

        for i,j in self._metrics.items():
            p = self._metrics[i].copy()
            p[p==0] = np.nan
            self._fuzzy_graph.vs()[i] = np.nanmean(p,axis=0)

        self._fuzzy_graph.vs()["name"] = range(self._metrics["stability"].shape[1])