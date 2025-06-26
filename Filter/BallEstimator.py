"""
estimate Ball Positions and Velocities from
ParticleSet.
"""
import numpy as np

from .ParticleSet import ParticleSet
from sklearn.cluster import KMeans # type: ignore

class BallEstimator:

    def __init__(self):
        pass

    def estimate(self, N: int, particle_set: ParticleSet) -> list[np.ndarray]:
        """
        Estimate N ball states from particle set.

        We use a weighted K-Means approach to estimate the ball positions (modes)
        from the multimodal density represented by the particles.

        (Why not GMM? -> thats slower than this, and preliminary testing did not
         show improved results)

        Inside each cluster, we then use a weighted average to get
        the ball velocities.
        """
        X = np.array(particle_set.particles)
        w = np.array(particle_set.weights)

        # clustering only positions proved more robust
        clf = KMeans(n_clusters = N, random_state = 0, n_init="auto").fit(X[:,:2], sample_weight = w)

        labels = clf.labels_
        positions = clf.cluster_centers_

        velocities = X[:, 2:]

        # velocity is average velocity in cluster
        output = []
        for (idx, p) in enumerate(positions):
            weights = w[labels == idx]
            weights = weights / np.sum(weights)
            v = np.sum(velocities[labels == idx] * weights.reshape(-1,1), axis=0)
            output.append(np.concatenate((p, v)))
        
        return output
