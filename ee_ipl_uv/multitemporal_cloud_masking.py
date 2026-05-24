"""
Created on June 12, 2016

@author: Gonzalo Mateo Garcia
@contact: gonzalo.mateo-garcia@uv.es
"""

import logging

from ee_ipl_uv import clustering

logger = logging.getLogger(__name__)

PARAMS_CLOUDCLUSTERSCORE_DEFAULT = {
    "threshold_cc": 5,
    "sampling_factor": 0.05,
    "lmbda": 1e-6,
    "gamma": 0.01,
    "threshold_dif_cloud": 0.04,
    "threshold_reflectance": 0.175,
    "do_clustering": True,
    "n_clusters": 10,
    "growing_ratio": 2,
}

def LocalCloudClusterScore(image_dict, background_dict, params=None):
    """
    The fully migrated, GEE-free main orchestrator function.
    Receives pre-loaded NumPy dictionaries directly from the local data loader.

    :param image_dict: Dictionary containing NumPy arrays for the target image {"B1": array, ...}
    :param background_dict: Dictionary containing NumPy arrays for the background prediction
    :param params: Parameter dictionary for thresholds and cluster count
    :return: Local NumPy cloud mask array (0: clear, 1: shadow, 2: cloud)
    """
    if params is None:
        params = dict(PARAMS_CLOUDCLUSTERSCORE_DEFAULT)

    local_mask = clustering.ClusterClouds(
        image=image_dict,
        background_prediction=background_dict,
        threshold_dif_cloud=params["threshold_dif_cloud"],
        do_clustering=params["do_clustering"],
        threshold_reflectance=params["threshold_reflectance"],
        bands_thresholds=["B2", "B3", "B4"],
        bands_clustering=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B10"],  # "B9" and "B11" excluded
        growing_ratio=params["growing_ratio"],
        n_clusters=params["n_clusters"],
        region_of_interest=None,
    )

    return local_mask