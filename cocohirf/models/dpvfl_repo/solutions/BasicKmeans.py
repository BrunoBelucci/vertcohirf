from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging

from util.eval_centers import eval_centers
from util.save_results import save_result_to_json


class BasicKmeans(KMeans):
    def __init__(self, config, tag='', save_result=False, **kwargs):
        self.k = config['k']
        self.tag = tag
        self.config = config
        self.save_result = save_result
        super().__init__(
            n_clusters=self.k,
        )

    def fit(self, data):
        if isinstance(data, list):
            assert len(data) == 1
            data = data[0]
        super().fit(data)
        labels = self.labels_
        losses = eval_centers(data, self.cluster_centers_)
        logging.info(f"local kmeans loss: {losses}")
        if self.save_result:
            silhouette = None
            # silhouette = silhouette_score(data, labels, metric='euclidean')
            logging.info(f"silhouette score with {self.k} centers is {silhouette}")
            self.save_results(losses, silhouette)
        return self.cluster_centers_

    def save_results(self, losses, silhouette):
        results = {
            "config": self.config,
            "losses": losses,
            "final_centers": self.cluster_centers_,
            "silhouette": silhouette,
        }
        print("save results...")
        save_result_to_json(results, self.tag, experiment=self.config['dataset']+"_basic")