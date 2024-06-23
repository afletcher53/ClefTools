from collections import defaultdict
from typing import Dict, List, Set, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as distance
from Article import Article
from DataController import DataController
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans


class ActiveLearning:
    """
    A class for performing active learning on a dataset.
    """

    VALID_ACQUISITION_FUNCTIONS = [
        "uncertainty",
        "certainty",
        "random",
        "alternating",
        "diversity",
        "adaptive_certainty_uncertainty",
        "clustering",
        "minimum_variance"
    ]

    def __init__(
        self,
        data_controller: DataController,
        model: BaseEstimator = LogisticRegression(),
        acquisition_function: str = "uncertainty",
        num_queries: Union[int, None] = None,
        batch_size: int = 10,
        reveal_size: int = 1,
        seed: int = 0,
        seed_start_pool_size: int = 1,
        seed_pool: List[str] = None,
        adaptive_threshold=0.8
    ):
        """
        Initialize the ActiveLearning object.

        :param model: The machine learning model to use
        :param dataset: The name of the dataset to use
        :param acquisition_function: The acquisition function for selecting samples
        :param num_queries: The number of queries to perform
        :param batch_size: The batch size for selecting samples
        :param reveal_size: The number of samples to reveal in each iteration
        :param seed: Random seed for reproducibility
        :param seed_start_pool_size: The size of the initial labeled pool
        """
        self.model = model
        self.dataset = data_controller
        self.batch_size = batch_size
        self.num_queries = num_queries
        self.acquisition_function = acquisition_function
        self.reveal_size = reveal_size
        self.seed = seed
        self.seed_start_pool_size = max(2, seed_start_pool_size)
        self.alternating_state = 0
        self.adaptive_threshold = adaptive_threshold
        self._validate_inputs()

        np.random.seed(self.seed)

        self.unlabeled: Dict[str, Article] = self.dataset.data.copy()
        self.labeled: Dict[str, Article] = {}
        self.iter: int = 0

        # Metrics
        self.n_N_history: List[float] = []
        self.percent_uncovered: List[float] = []

        self.seed_pool = seed_pool
        self._initialize_data()

    def _validate_inputs(self):
        """Validate input parameters."""
        if self.acquisition_function not in self.VALID_ACQUISITION_FUNCTIONS:
            raise ValueError(f"Invalid acquisition function. Choose from {
                             self.VALID_ACQUISITION_FUNCTIONS}")

        if self.batch_size > 0 and self.reveal_size > self.batch_size:
            raise ValueError("Reveal size cannot be greater than batch size.")

    def _initialize_data(self):
        """Initialize labeled and unlabeled data."""
        self._max_included = sum(
            1 for x in self.unlabeled.values() if x.label == 1)
        self._max_excluded = sum(
            1 for x in self.unlabeled.values() if x.label == 0)
        self.current_included = 0

        self._create_start_pool()

    def _create_start_pool(self):
        """Create the initial labeled pool."""
        if self.seed_pool:
            self.start_pool = self.seed_pool
        else:
            class_pmids = defaultdict(list)
            for pmid, article in self.unlabeled.items():
                class_pmids[article.label].append(pmid)

            self.start_pool = []
            for label in [0, 1]:
                if class_pmids[label]:
                    selected_pmid = np.random.choice(class_pmids[label])
                    self.start_pool.append(selected_pmid)
                    class_pmids[label].remove(selected_pmid)

            remaining_pmids = [pmid for pmids in class_pmids.values()
                               for pmid in pmids]
            additional_samples = min(
                len(remaining_pmids), self.seed_start_pool_size -
                len(self.start_pool)
            )
            if additional_samples > 0:
                self.start_pool.extend(
                    np.random.choice(
                        remaining_pmids, additional_samples, replace=False)
                )

        for pmid in self.start_pool:
            self.labeled[pmid] = self.unlabeled.pop(pmid)

    def run(self):
        """Run the active learning process until n/N reaches 100% or all relevant documents are found."""
        self.labeled_history = []  # Store labeled points at each iteration
        while True:
            self.iter += 1

            X, y = self.get_labelled_in_model_format()
            self.model.fit(X, y)

            indices = self.query()
            if len(indices) == 0:
                break

            percent_uncovered = len(self.labeled) / len(self.dataset.data)
            self.percent_uncovered.append(percent_uncovered)
            n_N = self.current_included / self.max_included
            self.n_N_history.append(n_N)

            self.update(indices)
            self.labeled_history.append(list(self.labeled.keys()))

            # Check if we've reached 100% n/N or found all relevant documents
            if n_N >= 1.0 or self.current_included == self.max_included:
                break

            # self.debug()

    def reset_data(self):
        """Reset the data to the initial state."""
        self.unlabeled = self.dataset.data.copy()
        self.labeled = {}
        self.iter = 0
        self.current_included = 0
        self.n_N_history = []
        self.percent_uncovered = []

    def debug(self):
        """Print debug information."""
        print(f"Iteration: {self.iter}")
        print(f"n/N: {self.n_N:.4f}")
        print(f"Percentage documents left: {
              (len(self.unlabeled) - len(self.labeled))/len(self.dataset.data):.2%}")
        print("-" * 40)

    def get_labelled_in_model_format(self):
        """Get labeled data in the format required by the model."""
        X = np.array(
            [article.input_vector for article in self.labeled.values()])
        y = np.array([article.label for article in self.labeled.values()])
        return X, y

    def get_unlabelled_in_model_format(self):
        """Get unlabeled data in the format required by the model."""
        X = np.array(
            [article.input_vector for article in self.unlabeled.values()])
        y = np.array([article.label for article in self.unlabeled.values()])
        return X, y

    def query(self):
        """Query for new samples to label."""
        X, y = self.get_unlabelled_in_model_format()

        if X.shape[0] == 0:
            return []

        if self.batch_size > 0:
            batch_size = min(self.batch_size, X.shape[0])
            batch_indices = np.random.choice(
                X.shape[0], batch_size, replace=False)
            X = X[batch_indices]
            y = y[batch_indices]
        else:
            batch_indices = None

        preds = self.model.predict_proba(X)

        if self.acquisition_function == "random":
            return self._random_sampling(X.shape[0], batch_indices)
        elif self.acquisition_function == "alternating":
            return self._alternating_sampling(preds, batch_indices)
        elif self.acquisition_function == "uncertainty":
            return self._uncertainty_sampling(preds, batch_indices)
        elif self.acquisition_function == "certainty":
            return self._certainty_sampling(preds, batch_indices)
        elif self.acquisition_function == "diversity":
            return self._diversity_sampling(X, batch_indices)
        elif self.acquisition_function == "adaptive_certainty_uncertainty":
            return self._adaptive_certainty_uncertainty_sampling(preds, batch_indices)
        elif self.acquisition_function == "clustering":
            return self._clustering_sampling(X, batch_indices)
        elif self.acquisition_function == "minimum_variance":
            return self._minimum_variance_sampling(X, batch_indices)
        else:
            raise ValueError(f"Acquisition function {
                             self.acquisition_function} not implemented.")

    def _minimum_variance_sampling(self, X, batch_indices):
        """
        Perform minimum variance sampling to keep feature variance as low as possible.
        """
        labeled_samples = self.get_labelled_in_model_format()[0]

        if len(labeled_samples) == 0:
            # If no samples are labeled yet, select the sample closest to the mean
            mean_sample = np.mean(X, axis=0)
            distances = cdist([mean_sample], X)[0]
            selected_index = np.argmin(distances)
            return [selected_index]

        current_variance = np.var(labeled_samples, axis=0)

        selected_indices = []
        remaining_indices = list(range(X.shape[0]))

        for _ in range(min(self.reveal_size, X.shape[0])):
            min_var_increase = float('inf')
            best_index = None

            for i in remaining_indices:
                new_samples = np.vstack([labeled_samples, X[i]])
                new_variance = np.var(new_samples, axis=0)
                var_increase = np.sum(new_variance - current_variance)

                if var_increase < min_var_increase:
                    min_var_increase = var_increase
                    best_index = i

            if best_index is not None:
                selected_indices.append(best_index)
                remaining_indices.remove(best_index)
                labeled_samples = np.vstack([labeled_samples, X[best_index]])
                current_variance = np.var(labeled_samples, axis=0)

        if batch_indices is not None:
            return batch_indices[selected_indices]
        else:
            return selected_indices

    def _uncertainty_sampling(self, preds, batch_indices):
        """Perform uncertainty sampling."""
        if preds.shape[1] != 2:
            raise ValueError(
                "Uncertainty sampling is only implemented for binary classification."
            )

        uncertainty = np.abs(preds[:, 0] - preds[:, 1])
        num_to_select = min(self.reveal_size, len(uncertainty))
        selected_indices = np.argsort(uncertainty)[-num_to_select:]

        if batch_indices is not None:
            return batch_indices[selected_indices]
        else:
            return selected_indices

    def _clustering_sampling(self, X, batch_indices):
        """
        Perform clustering-based sampling to keep feature variance low.
        """
        num_clusters = min(self.reveal_size, X.shape[0])
        kmeans = KMeans(n_clusters=num_clusters, random_state=self.seed)
        cluster_labels = kmeans.fit_predict(X)

        selected_indices = []
        for i in range(num_clusters):
            cluster_samples = np.where(cluster_labels == i)[0]
            if len(cluster_samples) > 0:
                selected_index = np.random.choice(cluster_samples)
                selected_indices.append(selected_index)

        if batch_indices is not None:
            return batch_indices[selected_indices]
        else:
            return selected_indices

    def _certainty_sampling(self, preds, batch_indices):
        """Perform certainty sampling."""
        if preds.shape[1] != 2:
            raise ValueError(
                "Certainty sampling is only implemented for binary classification."
            )

        certainty = np.abs(preds[:, 0] - preds[:, 1])
        num_to_select = min(self.reveal_size, len(certainty))
        selected_indices = np.argsort(certainty)[:num_to_select]

        if batch_indices is not None:
            return batch_indices[selected_indices]
        else:
            return selected_indices

    def _random_sampling(self, num_samples, batch_indices):
        """Perform random sampling."""
        num_to_select = min(self.reveal_size, num_samples)
        selected_indices = np.random.choice(
            num_samples, num_to_select, replace=False)

        if batch_indices is not None:
            return batch_indices[selected_indices]
        else:
            return selected_indices

    def _alternating_sampling(self, preds, batch_indices):
        """Perform alternating sampling between uncertainty and certainty."""
        if preds.shape[1] != 2:
            raise ValueError(
                "Alternating sampling is only implemented for binary classification."
            )

        if self.alternating_state == 0:
            # Uncertainty sampling
            uncertainty = np.abs(preds[:, 0] - preds[:, 1])
            num_to_select = min(self.reveal_size, len(uncertainty))
            selected_indices = np.argsort(uncertainty)[-num_to_select:]
        else:
            # Certainty sampling
            certainty = np.abs(preds[:, 0] - preds[:, 1])
            num_to_select = min(self.reveal_size, len(certainty))
            selected_indices = np.argsort(certainty)[:num_to_select]

        # Toggle the state for the next iteration
        self.alternating_state = 1 - self.alternating_state

        if batch_indices is not None:
            return batch_indices[selected_indices]
        else:
            return selected_indices

    def _adaptive_certainty_uncertainty_sampling(self, preds, batch_indices):
        """
        Perform adaptive sampling that switches from certainty to uncertainty
        once n/N reaches the specified threshold.
        """
        if preds.shape[1] != 2:
            raise ValueError(
                "Adaptive sampling is only implemented for binary classification.")

        current_n_N = self.current_included / self.max_included

        if current_n_N < self.adaptive_threshold:
            # Use certainty sampling
            certainty = np.abs(preds[:, 0] - preds[:, 1])
            num_to_select = min(self.reveal_size, len(certainty))
            selected_indices = np.argsort(certainty)[:num_to_select]
        else:
            # Switch to uncertainty sampling
            uncertainty = np.abs(preds[:, 0] - preds[:, 1])
            num_to_select = min(self.reveal_size, len(uncertainty))
            selected_indices = np.argsort(uncertainty)[-num_to_select:]

        if batch_indices is not None:
            return batch_indices[selected_indices]
        else:
            return selected_indices

    def _diversity_sampling(self, X, batch_indices):
        labeled_samples = self.get_labelled_in_model_format()[0]
        distances = distance.cdist(X, labeled_samples, metric="euclidean")
        diversity_scores = np.min(distances, axis=1)
        num_to_select = min(self.reveal_size, len(diversity_scores))
        selected_indices = np.argsort(diversity_scores)[-num_to_select:]

        if batch_indices is not None:
            return batch_indices[selected_indices]
        else:
            return selected_indices

    def update(self, selected_indices):
        """Update the labeled and unlabeled sets based on the selected indices."""

        selected_pmids = [list(self.unlabeled.keys())[i]
                          for i in selected_indices]
        for pmid in selected_pmids:
            self.labeled[pmid] = self.unlabeled.pop(pmid)
            self.current_included += self.labeled[pmid].label

    @property
    def max_included(self):
        """Get the maximum number of included articles."""
        return self._max_included

    @property
    def max_excluded(self):
        """Get the maximum number of excluded articles."""
        return self._max_excluded

    @property
    def n_N(self):
        """Get the ratio of included articles so far (n/N)."""
        return self.current_included / self.max_included

    @property
    def documents_uncovered(self):
        """Get the total documents uncovered at each iteration."""
        return len(self.labeled)


def generate_seed_pools(
    dataset, num_pools: int, seed_start_pool_size: int
) -> List[List[str]]:
    """Generate multiple unique seed pools, ensuring each pool has at least one sample from each class."""
    all_pmids = list(dataset.data.keys())
    class_0_pmids = [
        pmid for pmid, article in dataset.data.items() if article.label == 0
    ]
    class_1_pmids = [
        pmid for pmid, article in dataset.data.items() if article.label == 1
    ]

    seed_pools: Set[tuple] = set()
    attempts = 0
    max_attempts = num_pools * 10

    while len(seed_pools) < num_pools and attempts < max_attempts:
        seed_pool = []
        seed_pool.append(np.random.choice(class_0_pmids))
        seed_pool.append(np.random.choice(class_1_pmids))
        remaining_size = seed_start_pool_size - 2
        if remaining_size > 0:
            remaining_pmids = list(set(all_pmids) - set(seed_pool))
            seed_pool.extend(
                np.random.choice(
                    remaining_pmids, remaining_size, replace=False)
            )

        seed_pool.sort()
        seed_pool_tuple = tuple(seed_pool)

        if seed_pool_tuple not in seed_pools:
            seed_pools.add(seed_pool_tuple)

        attempts += 1

    if len(seed_pools) < num_pools:
        print(f"Warning: Only generated {
              len(seed_pools)} unique seed pools out of {num_pools} requested.")

    return [list(pool) for pool in seed_pools]


def animate_clustering(
    reduced_features, all_labels, active_learner, baseline=False, interval=500
):
    fig, ax = plt.subplots(figsize=(10, 10))

    scatter = ax.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=all_labels,
        alpha=0.1,
        cmap="coolwarm",
    )
    labeled_scatter = ax.scatter(
        [], [], c=[], s=50, edgecolor="black", linewidth=1, cmap="coolwarm"
    )

    ax.set_xlabel("t-SNE component 1")
    ax.set_ylabel("t-SNE component 2")
    ax.set_title("Feature Space Clustering")

    plt.colorbar(scatter, ax=ax, label="Class")
    ax.legend(["All data", "Labeled data"])

    def update(frame):
        if baseline:
            labeled_indices = [
                list(active_learner.dataset.data.keys()).index(pmid)
                for pmid in active_learner.baseline_labeled_history[frame]
            ]
            title_prefix = "Baseline"
        else:
            labeled_indices = [
                list(active_learner.dataset.data.keys()).index(pmid)
                for pmid in active_learner.labeled_history[frame]
            ]
            title_prefix = "Active Learning"

        labeled_features = reduced_features[labeled_indices]
        labeled_labels = all_labels[labeled_indices]

        labeled_scatter.set_offsets(labeled_features)
        labeled_scatter.set_array(labeled_labels)
        ax.set_title(
            f"{title_prefix} Feature Space Clustering - Iteration {frame + 1}")
        return (labeled_scatter,)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(
            active_learner.baseline_labeled_history
            if baseline
            else active_learner.labeled_history
        ),
        interval=interval,
        blit=True,
    )
    return anim


def run_and_plot_active_learning(dataset, batch_size=50, reveal_size=25, seed=0, num_runs=1, seed_start_pool_size=2, generate_clustering=False, adaptive_threshold=0.8):
    model = LogisticRegression()
    dataset_controller = DataController(dataset)
    seed_pools = generate_seed_pools(
        dataset_controller, num_runs, seed_start_pool_size)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    if generate_clustering:
        all_features = np.array(
            [article.input_vector for article in dataset_controller.data.values()])
        all_labels = np.array(
            [article.label for article in dataset_controller.data.values()])
        tsne = TSNE(n_components=2, random_state=seed)
        reduced_features = tsne.fit_transform(all_features)

    acquisition_functions = ["uncertainty", "certainty", "random", "alternating",
                             "diversity", "adaptive_certainty_uncertainty", "minimum_variance"]
    colors = ['blue', 'green', 'red', 'purple', 'orange',
              'brown', 'pink']  # Added a color for minimum_variance

    lines_dict = {acq_func: {'n_N': [], 'feature_coverage': []}
                  for acq_func in acquisition_functions}

    for i, seed_pool in enumerate(seed_pools):
        for j, acq_func in enumerate(acquisition_functions):
            active_learner = ActiveLearning(
                data_controller=dataset_controller,
                model=model,
                batch_size=batch_size,
                reveal_size=reveal_size,
                seed=seed + i,
                seed_start_pool_size=seed_start_pool_size,
                seed_pool=seed_pool,
                acquisition_function=acq_func,
                adaptive_threshold=adaptive_threshold
            )

            active_learner.run()

            n_N = [x * 100 for x in active_learner.n_N_history]
            percent_uncovered = [
                x * 100 for x in active_learner.percent_uncovered]
            feature_coverage = calculate_feature_coverage_variance(
                active_learner)

            lines_dict[acq_func]['n_N'].append((percent_uncovered, n_N))
            lines_dict[acq_func]['feature_coverage'].append(
                (percent_uncovered, feature_coverage))

            if generate_clustering and i == num_runs - 1:
                ax_cluster = fig.add_subplot(3, 1, j+1)
                plot_clustering(ax_cluster, reduced_features, all_labels, active_learner,
                                title=f"Final Feature Space Clustering ({acq_func.capitalize()})")

            del active_learner

    def plot_confidence_interval(ax, x_data, y_data, color, label):
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        # Interpolate all runs to have the same x values
        max_len = max(len(x) for x in x_data)
        x_interp = np.linspace(0, 100, max_len)
        y_interp = np.array([np.interp(x_interp, x, y)
                            for x, y in zip(x_data, y_data)])

        mean = np.mean(y_interp, axis=0)
        ci = stats.t.interval(
            0.95, len(y_interp) - 1, loc=mean, scale=stats.sem(y_interp, axis=0)
        )

        ax.fill_between(x_interp, ci[0], ci[1], color=color, alpha=0.3)
        ax.plot(x_interp, mean, color=color, label=label)

    for j, acq_func in enumerate(acquisition_functions):
        x_data_n_N = [run[0] for run in lines_dict[acq_func]["n_N"]]
        y_data_n_N = [run[1] for run in lines_dict[acq_func]["n_N"]]
        plot_confidence_interval(
            ax1, x_data_n_N, y_data_n_N, colors[j], acq_func.capitalize()
        )

        x_data_fc = [run[0]
                     for run in lines_dict[acq_func]["feature_coverage"]]
        y_data_fc = [run[1]
                     for run in lines_dict[acq_func]["feature_coverage"]]
        plot_confidence_interval(
            ax2, x_data_fc, y_data_fc, colors[j], acq_func.capitalize()
        )

    ax1.set_xlabel("Documents Reviewed (%)")
    ax1.set_ylabel("n/N (%)")
    ax1.set_title(f"Active Learning Performance: {dataset} - {num_runs} runs")
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel("Documents Reviewed (%)")
    ax2.set_ylabel("Feature Space Coverage (%)")
    ax2.set_title("Feature Space Coverage Over Time")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"active_learning_{dataset}_{num_runs}_{
                reveal_size}_{batch_size}_{seed}_{adaptive_threshold}.png")
    plt.close(fig)

    del dataset_controller


def plot_clustering(
    ax,
    reduced_features,
    all_labels,
    active_learner,
    baseline=False,
    title="Feature Space Clustering",
):
    # Plot all points
    scatter = ax.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=all_labels,
        alpha=0.1,
        cmap="coolwarm",
    )

    # Highlight labeled points
    if baseline:
        labeled_indices = [
            list(active_learner.dataset.data.keys()).index(pmid)
            for pmid in active_learner.baseline_labeled_history[-1]
        ]
    else:
        labeled_indices = [
            list(active_learner.dataset.data.keys()).index(pmid)
            for pmid in active_learner.labeled.keys()
        ]

    labeled_features = reduced_features[labeled_indices]
    labeled_labels = all_labels[labeled_indices]

    ax.scatter(
        labeled_features[:, 0],
        labeled_features[:, 1],
        c=labeled_labels,
        s=50,
        edgecolor="black",
        linewidth=1,
        cmap="coolwarm",
    )

    plt.colorbar(scatter, ax=ax, label="Class")

    # Add a legend
    ax.legend(["All data", "Labeled data"])
    ax.set_title(title)


def calculate_feature_coverage_variance(active_learner):
    all_features = np.array(
        [article.input_vector for article in active_learner.dataset.data.values()]
    )
    feature_coverage = []
    for i in range(len(active_learner.n_N_history)):
        labeled_features = np.array(
            [
                article.input_vector
                for article in list(active_learner.labeled.values())[: i + 1]
            ]
        )

        if labeled_features.shape[0] == 0:
            feature_coverage.append(0)
            continue

        all_variance = np.var(all_features, axis=0)
        labeled_variance = np.var(labeled_features, axis=0)
        all_variance = np.where(all_variance == 0, 1e-10, all_variance)
        labeled_variance = np.where(
            labeled_variance == 0, 1e-10, labeled_variance)
        variance_ratio = labeled_variance / all_variance
        average_ratio = np.mean(variance_ratio)

        coverage_percentage = min(average_ratio * 100, 100)

        feature_coverage.append(coverage_percentage)

    return feature_coverage


# Usage example
if __name__ == "__main__":
    run_and_plot_active_learning(
        "CD010438",
        num_runs=2,
        batch_size=75,
        reveal_size=25,
        seed_start_pool_size=4,
        generate_clustering=False,
        adaptive_threshold=0.8
    )
