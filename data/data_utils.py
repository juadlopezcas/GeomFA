import numpy as np
from sklearn.preprocessing import scale
from sklearn.datasets import make_blobs
from scipy.stats import expon, special_ortho_group
from sklearn.datasets import fetch_olivetti_faces
from numpy.random import RandomState
import logging
import cv2


class DataGenerator:
    def __init__(self, num_samples, num_dimensions):
        self.num_samples = num_samples
        self.num_dimensions = num_dimensions
        self.matrix_axis = np.identity(self.num_dimensions)
        #self.matrix_axis = self.generate_orthonormal_matrix()
    
    def generate_orthonormal_matrix(self):
        return special_ortho_group.rvs(self.num_dimensions)
    
    def cross_data(self,):
        orthogonal_matrix = self.matrix_axis
        means = np.random.normal(0, 0.25,self.num_dimensions)
        covariances = np.diag(np.random.lognormal(0, 0.5, self.num_dimensions))
        samples = np.random.multivariate_normal(means, covariances, self.num_samples)
        m, n = samples.shape
        projected_data = []
        for i in range(n):
            column_vector = orthogonal_matrix[:, i]
            for j in range(m):
                projection_vector = np.dot(samples[j], column_vector)*column_vector
                projected_data.append(projection_vector)
        return np.concatenate((np.array(projected_data),0.6*samples))
    
    def gaussian_factor_analysis_data(self,):
        if self.num_dimensions == 1:
            # Parameters for the Gaussian components
            n_samples = self.num_samples
            u = 0.92
            covariances = [np.array([[1, u], [u, 1]]), np.array([[1, -u], [-u, 1]])]
            weights = [0.5, 0.5]  # Mixing probabilities

            # Generate samples
            samples = np.vstack([
                np.random.multivariate_normal(mean=[0, 0], cov=covariances[0], size=int(n_samples * weights[0])),
                np.random.multivariate_normal(mean=[0, 0], cov=covariances[1], size=int(n_samples * weights[1]))
            ])

            # Shuffle the samples to mix the two distributions
            np.random.shuffle(samples)
        else:
            # Parameters for the Gaussian components
            n_samples = self.num_samples
            t = 0.05
            d = self.num_dimensions
            k = d-1
            ## Full dimensional case
            covariances = [np.diag([1-t if i==j else t/(d-1) for i in range(1,d+1)]) for j in range(1,d+1)]
            weights = 1/d*np.ones(d)
            # Full dimensional samples
            samples = np.vstack([np.random.multivariate_normal(mean=np.zeros(d), cov=covariances[j], size=int(n_samples * weights[j])) for j in range(d)])
            #Rotate the samples for general crosses
            #samples = samples @ self.matrix_axis
            np.random.shuffle(samples)
        return samples
    
    def gaussian_top_k_data(self,):
        n_samples = self.num_samples
        t = 0.04
        d = self.num_dimensions
        k = d-1
        covariances = [np.diag([1-t if i==j else t/(d-1) for i in range(1,d+1)]) for j in range(1,k+1)]
        weights = 1/d*np.ones(d)
        k_samples = np.vstack([np.random.multivariate_normal(mean=np.zeros(d), cov=covariances[j], size=int(n_samples * weights[j])) for j in range(k)])
        covariances.append(np.diag([(1-t) if i==d else t/(d-1) for i in range(1,d+1)]))
        gaussian_samples = 0.6*np.random.multivariate_normal(mean=np.zeros(d), cov=covariances[-1], size=int(n_samples * 1/d))
        rot_inv_samples = 0.6*np.random.multivariate_normal(mean=np.zeros(d), cov=np.identity(d), size=int(n_samples * 1/d))
        #return np.vstack([k_samples, gaussian_samples])
        #return k_samples
        return np.vstack([k_samples, gaussian_samples, rot_inv_samples])

    def exponential_factor_analysis_data(self,):
        if self.num_dimensions == 2:
            # Parameters for the exponential components
            n = self.num_samples
            norm_dat = np.random.normal(size=(n, 2))
            norm_dat = scale(norm_dat)

            exp_dat = np.random.choice([-1, 1], size=(2 * n, 2)) * expon.rvs(size=(2 * n, 2)) ** 1.3
            exp_dat = scale(exp_dat)
            #s = np.linalg.svd(self.matrix_axis)
            s = np.linalg.svd([[1, -2], [-3, 1]])
            R = s[0]
            rexp_dat = np.dot(exp_dat, R)
        if self.num_dimensions == 3:
            # Parameters for the exponential components
            n = self.num_samples
            norm_dat = np.random.normal(size=(n, 3))
            norm_dat = scale(norm_dat)

            exp_dat = np.random.choice([-1, 1], size=(2 * n, 3)) * expon.rvs(size=(2 * n, 3)) ** 1.3
            exp_dat = scale(exp_dat)

            s = np.linalg.svd([[1, -2, 1], [-3, 1, 1], [1, 1, -1]])
            R = s[0]
            rexp_dat = np.dot(exp_dat, R)
        return rexp_dat
    
    def generate_and_align_clusters(self, cluster_std=1.0, centering = True):
        # Generate random clusters
        X, y = make_blobs(n_samples=self.num_samples, n_features=self.num_dimensions, centers=self.num_dimensions, cluster_std=cluster_std)

        if centering:
            # Calculate new centers for clusters along the axes
            new_centers = np.identity(self.num_dimensions)
            # Calculate translation vectors for each cluster
            old_centers = np.array([X[y == i].mean(axis=0) for i in range(self.num_dimensions)])
            translation_vectors = new_centers - old_centers

            # Translate the points to align the cluster centers to the new centers
            for i in range(self.num_dimensions):
                X[y == i] += translation_vectors[i]
        return X
    
class DataLoader:
    def __init__(self):
        self.data = self.faces()

    def faces(self,size=32):
        rng = RandomState(0)
        # Display progress logs on stdout
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng)
        images_resampled = self.resample_flattened_images(faces, original_size=64, new_size=size)
        n_samples, n_features = images_resampled.shape
        faces_centered = images_resampled - images_resampled.mean(axis=0)
        faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)
        return faces_centered

    def resample_flattened_images(images, original_size=64, new_size=32):
        num_images = images.shape[0]
        # Reshape to (num_images, original_size, original_size)
        images_reshaped = images.reshape((num_images, original_size, original_size))
        # Resample each image to (new_size, new_size)
        resampled_images = np.array([
            cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_AREA)
        for img in images_reshaped
        ])
        # Flatten the resampled images back to vectors of length new_size * new_size
        resampled_images_flattened = resampled_images.reshape((num_images, new_size * new_size))
        return resampled_images_flattened

def run():
    data_generator = DataGenerator(1000, 3)
    return data_generator

if __name__ == "__main__":
    run()
