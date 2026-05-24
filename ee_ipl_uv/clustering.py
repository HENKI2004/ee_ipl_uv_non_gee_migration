import numpy as np
from scipy.ndimage import binary_opening
from sklearn.cluster import KMeans

BANDS_MODEL = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"]


def SelectClusters(
    image,
    background_prediction,
    result_clustering,
    n_clusters,
    bands_thresholds=["B2", "B3", "B4"],
    region_of_interest=None,
):
    """
    Function that contains the logic to create the cluster score mask. given the clustering result.

    :param image:
    :param background_prediction:
    :param result_clustering:
    :param n_clusters:
    :param region_of_interest:
    :return:
    """

    if isinstance(result_clustering, np.ndarray):
        diff_dict = {}
        for b in bands_thresholds:
            diff_dict[b] = (
                image[b] - background_prediction[b]
            )  # bands_norm_difference = [b+"_difference" for b in bands_thresholds]

        multitemporal_score = np.zeros_like(result_clustering, dtype=np.float32)
        reflectance_score = np.zeros_like(result_clustering, dtype=np.float32)

        for i in range(n_clusters):
            cluster_mask = result_clustering == i
            if not np.any(cluster_mask):
                continue

            refl_values = [image[b][cluster_mask] for b in bands_thresholds]
            refl_stack = np.stack(refl_values, axis=0)

            clusteri_refl_norm = np.sqrt(np.mean(refl_stack**2))

            diff_values = [diff_dict[b][cluster_mask] for b in bands_thresholds]
            diff_stack = np.stack(diff_values, axis=0)

            clusteridiff_mean = np.mean(diff_stack)
            clusteridiff_norm = np.sqrt(np.mean(diff_stack**2))

            if clusteridiff_mean > 0:
                multitemporal_score_clusteri = clusteridiff_norm
            else:
                multitemporal_score_clusteri = clusteridiff_norm * -1

            multitemporal_score[cluster_mask] = multitemporal_score_clusteri
            reflectance_score[cluster_mask] = clusteri_refl_norm

        return multitemporal_score, reflectance_score


def ClusterClouds(
    image,
    background_prediction,
    threshold_dif_cloud=0.045,
    do_clustering=True,
    numPixels=1000,
    threshold_reflectance=0.175,
    bands_thresholds=["B2", "B3", "B4"],
    bands_clustering=BANDS_MODEL,
    growing_ratio=2,
    n_clusters=10,
    region_of_interest=None,
):
    """
    Function that compute the cloud score given the differences between the real and predicted image.

    :param image:
    :param background_prediction: image_real - image_pred
    :param threshold_dif_cloud: Threshold over the cloud score to be considered clouds
    :param threshold_reflectance: Threshold over the cloud score to be considered shadows
    :param do_clustering: Wether to do the clustering or not
    :param n_clusters: number of clusters
    :param bands_thresholds: Bands used to set the thresholds
    :param numPixels:  to be considered by the clustering algorithm
    :param region_of_interest:  region of interest within the image
    :return: ee.Image with 0 for clear pixels, 1 for shadow pixels and 2 for cloudy pixels
    """

    if isinstance(image, dict):
        img_differences = {
            b: image[b] - background_prediction[b] for b in bands_clustering
        }

        if do_clustering:
            h, w = image[bands_clustering[0]].shape
            bands_data = [img_differences[b].ravel() for b in bands_clustering]
            X = np.stack(bands_data, axis=-1)

            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0) + 1e-6
            X_normalized = (X - X_mean) / X_std

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            flat_labels = kmeans.fit_predict(X_normalized)
            result = flat_labels.reshape(h, w)

            multitemporal_score, reflectance_score = SelectClusters(
                image,
                background_prediction,
                result,
                n_clusters,
                bands_thresholds,
                region_of_interest,
            )
        else:
            diff_thresh_data = np.stack(
                [image[b] - background_prediction[b] for b in bands_thresholds], axis=0
            )
            refl_thresh_data = np.stack([image[b] for b in bands_thresholds], axis=0)

            arrayImageDiffmean = np.mean(diff_thresh_data, axis=0)
            arrayImageDiffnorm = np.sqrt(np.mean(diff_thresh_data**2, axis=0))
            reflectance_score = np.sqrt(np.mean(refl_thresh_data**2, axis=0))

            multitemporal_score = arrayImageDiffnorm * (arrayImageDiffmean >= 0)

        if threshold_reflectance <= 0:
            cloud_score_threshold = multitemporal_score > threshold_dif_cloud
        else:
            cloud_score_threshold = (multitemporal_score > threshold_dif_cloud) & (
                reflectance_score > threshold_reflectance
            )

        y, x = np.ogrid[
            -growing_ratio : growing_ratio + 1, -growing_ratio : growing_ratio + 1
        ]
        kernel = x**2 + y**2 <= growing_ratio**2

        final_local_mask = binary_opening(
            cloud_score_threshold, structure=kernel
        ).astype(np.uint8)

        return final_local_mask
