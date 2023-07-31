"""Functions to cluster structures.

functions:
get_rmsd: Computes the rmsd matrix of the conformers in a list of rdkit mol
    objects.
get_labels_affty: Clusters data in affinity matrix form by assigning labels to
    data points.
get_labels_vector: Clusters data in vectorial form by assigning labels to
    data points.
get_clusters: Groups data-points belonging to the same cluster into arrays of
    indices.
get_exemplars_affty: Computes the exemplars for every cluster and returns a list
    of indices.
plot_clusters: Plots the clustered data casting a color to every cluster.
clustering: Directs the clustering process by calling the relevant functions.
"""
import logging

import hdbscan
import numpy as np

logger = logging.getLogger('DockOnSurf')


def get_rmsd(mol_list: list, remove_Hs="c"):
    """Computes the rmsd matrix of the conformers in a list of rdkit mol objects

    @param mol_list: list of rdkit mol objects containing the conformers.
    @param remove_Hs: bool or str,
    @return rmsd_matrix: Matrix containing the rmsd values of every pair of
    conformers.

    The RMSD values of every pair of conformers is computed, stored in matrix
    form and returned back. The calculation of rmsd values can take into
    account all hydrogens, none, or only the ones not linked to carbon atoms.
    """
    import rdkit.Chem.AllChem as Chem

    if len(mol_list) < 2:
        err = "The provided molecule has less than 2 conformers"
        logger.error(err)
        raise ValueError(err)

    if not remove_Hs:
        pass
    elif remove_Hs or remove_Hs.lower() == "all":
        mol_list = [Chem.RemoveHs(mol) for mol in mol_list]
    elif remove_Hs.lower() == "c":
        from src.dockonsurf.isolated import remove_C_linked_Hs
        mol_list = [remove_C_linked_Hs(mol) for mol in mol_list]
    else:
        err = "remove_Hs value does not have an acceptable value"
        logger.error(err)
        raise ValueError(err)

    num_confs = len(mol_list)
    conf_ids = list(range(num_confs))
    rmsd_mtx = np.zeros((num_confs, num_confs))
    for id1 in conf_ids:  # TODO reduce RMSD precision
        for id2 in conf_ids[id1 + 1:]:
            rmsd = Chem.GetBestRMS(mol_list[id1], mol_list[id2])
            rmsd_mtx[id1][id2] = rmsd
            rmsd_mtx[id2][id1] = rmsd

    return rmsd_mtx


def get_labels_affty(affty_mtx, kind="rmsd"):
    """Clusters data in affinity matrix form by assigning labels to data points.

    @param affty_mtx: Data to be clustered, it must be an affinity matrix.
    (Eg. Euclidean distances between points, RMSD Matrix, etc.).
    Shape: [n_points, n_points]
    @param kind: Which kind of data the affinity matrix contains.
    @return: list of cluster labels. Every data point is assigned a number
    corresponding to the cluster it belongs to.
    """
    if np.average(affty_mtx) < 1e-3 and kind == "rmsd":
        sing_clust = True
        min_size = int(len(affty_mtx) / 2)
    else:
        sing_clust = False
        min_size = 20
    hdbs = hdbscan.HDBSCAN(metric="precomputed",
                           min_samples=5,
                           min_cluster_size=min_size,
                           allow_single_cluster=sing_clust)
    return hdbs.fit_predict(affty_mtx)


def get_labels_vector():
    """Clusters data in vectorial form by assigning labels to data points.

    @return: list of cluster labels. Every data point is assigned a number
    corresponding to the cluster it belongs to.
    """
    # TODO Implement it.
    return []


def get_clusters(labels):
    """Groups data-points belonging to the same cluster into arrays of indices.

    @param labels: list of cluster labels (numbers) corresponding to the cluster
    it belongs to.
    @return: tuple of arrays. Every array contains the indices (relative to the
    labels list) of the data points belonging to the same cluster.
    """
    n_clusters = max(labels) + 1
    return tuple(np.where(labels == clust_num)[0]
                 for clust_num in range(n_clusters))


def get_exemplars_affty(affty_mtx, clusters):
    """Computes the exemplars for every cluster and returns a list of indices.

    @param affty_mtx: Data structured in form of affinity matrix. eg. Euclidean
    distances between points, RMSD Matrix, etc.) shape: [n_points, n_points].
    @param clusters: tuple of arrays. Every array contains the indices (relative
    to the affinity matrix) of the data points belonging to the same cluster.
    @return: list of indices (relative to the affinity matrix) of the exemplars
    for every cluster.

    This function finds the exemplars of already clusterized data. It does
    that by (i) building a rmsd matrix for each existing cluster with the values
    of the total RMSD matrix (ii) carrying out an actual clustering for each
    cluster-specific matrix using a set of parameters (large negative value of
    preference) such that it always finds only one cluster and (iii) it then
    calculates the exemplar for the matrix.
    """
    from sklearn.cluster import AffinityPropagation
    # Splits Total RMSD matrix into cluster-specific RMSD matrices.
    clust_affty_mtcs = tuple(affty_mtx[np.ix_(clust, clust)]
                             for clust in clusters)
    exemplars = []
    # Carries out the forced-to-converge-to-1 clustering for each already
    # existing cluster rmsd matrix and calculates the exemplar.
    for i, mtx in enumerate(clust_affty_mtcs):
        pref = -1e6 * np.max(np.abs(mtx))
        af = AffinityPropagation(affinity='precomputed', preference=pref,
                                 damping=0.95, max_iter=2000,
                                 random_state=None).fit(mtx)
        exemplars.append(clusters[i][af.cluster_centers_indices_[0]])
    return exemplars


def plot_clusters(labels, x, y, exemplars=None, save=True):
    """Plots the clustered data casting a color to every cluster.

    @param labels: list of cluster labels (numbers) corresponding to the cluster
    it belongs to.
    @param x: list of data of the x axis.
    @param y: list of data of the y axis.
    @param exemplars: list of data point indices (relative to the labels list)
    considered as cluster exemplars.
    @param save: bool, Whether to save the generated plot into a file or not.
    (in the latter case the plot is shown in a new window)
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    n_clusters = max(labels) + 1
    rb = cm.get_cmap('gist_rainbow', max(n_clusters, 1))
    rb.set_under()
    plt.figure(figsize=(10, 8))
    for i in range(len(labels)):
        plt.plot(x[i], y[i], c=rb(labels[i]), marker='.')
        if len(exemplars) > 0 and i == exemplars[labels[i]]:
            plt.plot(x[i], y[i], c=rb(labels[i]), marker="x",
                     markersize=15,
                     label=f"Exemplar cluster {labels[i]}")
    plt.title(f'Found {n_clusters} Clusters.')
    plt.xlabel("Energy")
    plt.ylabel("MOI")
    plt.legend()

    bounds = list(range(max(n_clusters, 1)))
    norm = colors.Normalize(vmin=min(labels), vmax=max(labels))
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=rb), ticks=bounds)
    if save:
        from src.dockonsurf.utilities import check_bak
        check_bak('clusters.png')
        plt.savefig('clusters.png')
        plt.close("all")
    else:
        plt.show()


def clustering(data, plot=False, x=None, y=None):
    """Directs the clustering process by calling the relevant functions.

    @param data: The data to be clustered. It must be stored in vector form
    [n_features, n_samples] or in affinity matrix form [n_samples, n_samples],
    symmetric and 0 in the main diagonal. (Eg. Euclidean distances between
    points, RMSD Matrix, etc.).
    @param plot: bool, Whether to plot the clustered data.
    @param x: Necessary only if plot is turned on. X values to plot the data.
    @param y: Necessary only if plot is turned on. Y values to plot the data.
    @return: list of exemplars, list of indices (relative to data)
    exemplars for every cluster.
    """
    from collections.abc import Iterable

    data_err = "Data must be stored in vector form [n_features, n_samples] or" \
               "in affinity matrix form [n_samples, n_samples]: symmetric " \
               "and 0 in the main diagonal. Eg. RMSD matrix"
    debug_err = "On debug mode x and y should be provided"

    if plot and not (isinstance(x, Iterable) and isinstance(y, Iterable)):
        logger.error(debug_err)
        raise ValueError(debug_err)
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if len(data.shape) != 2:
        logger.error(data_err)
        raise ValueError(data_err)

    if data.shape[0] == data.shape[1] \
            and (np.tril(data).T == np.triu(data)).all():
        logger.info("Clustering using affinity matrix.")
        labels = get_labels_affty(data)
        if max(labels) == -1:
            logger.warning('Clustering of conformers did not converge. Try '
                           "setting a smaller 'min_samples' parameter.")
            exemplars = list(range(data.shape[0]))
        else:
            clusters = get_clusters(labels)
            exemplars = get_exemplars_affty(data, clusters)
        if plot:
            plot_clusters(labels, x, y, exemplars, save=True)
        logger.info(f'Conformers are grouped in {len(exemplars)} clusters.')
        return exemplars
    else:
        not_impl_err = 'Clustering not yet implemented for vectorized data'
        logger.error(not_impl_err)
        raise NotImplementedError(not_impl_err)
