import numpy as np
from sklearn.decomposition import PCA, FastICA, NMF, TruncatedSVD



class FeatureExtractor:

    """
    Object for performing feature extraction / dimensionality reduction on input data. Accept either names of method from scikit-learn or custom feature extractors.

    Parameters
    ----------
    method : str | object
        If str, supported methods are 'pca', 'ica' and 'nmf', from the scikit-learn library. If object, the requirements are:
        - the output has the same first dimension as input data.
        - the object has either a 'fit_predict', 'predict', 'fit_transform' or 'transform' method.
    params : dict
        Dictionary containing the arguments to be used for the initialization of the method.
    """


    def __init__(self, method, params={}):

        self.method = method

        if method == 'pca':
            self.model = PCA(**params)

        elif method == 'ica':
            self.model = ICA(**params)

        elif method == 'nmf':
            self.model = NMF(**params)

        elif hasattr(method, '__dict__'):
            assert(callable(getattr(method, 'fit_predict', None)) or callable(getattr(method, 'predict', None)))
            self.model = method(**params) if callable(method) else method

        else:
            raise NameError(F'Method \"{method}\" was not recognized. Please choose between: pca, ica, nmf or provide a custom extractor (see doc).')


    def visualize(self, data):

        """
        Method for visualizing the quality of the dimensionality reduction. Currently only implemented for PCA (plot of the cumulated explained variance ratio).

        Parameters
        ----------
        data : ndarray
            2D data with samples as lines and features as columns
        """

        if self.method == 'pca':

            vis_model = PCA(n_components=data.shape[1])
            vis_model.fit(data)

            res = []
            for i in range(vis_model.explained_variance_ratio_):
                res.append(np.sum(var[:i]))

            plt.ylabel("Cumulated explained variance ratio")
            plt.xticks([5*i for i in range(int(data.shape[-1]/5+1))])
            plt.xlabel("Number of principal components")
            plt.vlines(data.shape[1], 0, 1, color='gray', linestyle='dashed')
            plt.hlines(0.5, 0, 45, color='gray', linestyle='dashed')
            plt.xlim(0, data.shape[1])
            plt.ylim(0, 1)
            plt.plot(res)
            plt.show()

        else:

            raise(NotImplementedError, 'Visualization has been implemented for the selected method.')


    def process(self, data):

        """
        Method for performing the feature extraction on input data.

        Parameters
        ----------
        data : ndarray
            2D data with samples as lines and features as columns.

        Returns
        -------
        res : ndarray
            2D array containing the reduced data, with samples as lines and features as columns.
        """

        if self.method == 'nmf':
            data = (data - np.min(data)) / (np.max(data) - np.min(data))

        if hasattr(self.model, 'fit_predict') and callable(getattr(self.model, 'fit_predict')):
            res = self.model.fit_predict(data)

        elif hasattr(self.model, 'fit_transform') and callable(getattr(self.model, 'fit_transform')):
            res = self.model.fit_transform(data)

        elif hasattr(self.model, 'predict') and callable(getattr(self.model, 'predict')):
            res = self.model.predict(data)

        elif hasattr(self.model, 'transform') and callable(getattr(self.model, 'transform')):
            res = self.model.predict(data)

        else:
            pass

        return(res)
