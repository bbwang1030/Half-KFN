"""The classical non-differentiable Friedman-Rafsky and k-NN tests."""
import math

from scipy.sparse.csgraph import minimum_spanning_tree as mst
from torch.autograd import Function
import numpy as np
import torch
from .permutation_test import permutation_test_mat,permutation_test_mat_KFN,permutation_test_mat_halfKFN,permutation_test_mat_halfKFN_validation
from .util import pdist,pdist_within_class,pdist_within_class_one_dimension,each_class_sample
import scipy
import csv
from itertools import permutations
__all__ = ['FRStatistic', 'KNNStatistic','SmoothKNNStatistic','KFNStatistic','halfKFNStatistic']


class MSTFn(Function):
    """Compute the minimum spanning tree given a matrix of pairwise weights."""

    @staticmethod
    def forward(ctx, weights):
        """Compute the MST given the edge weights.

        The behaviour is the same as that of ``minimum_spanning_tree` in
        ``scipy.sparse.csgraph``, namely i) the edges are assumed non-negative,
        ii) if ``weights[i, j]`` and ``weights[j, i]`` are both non-negative,
        their minimum is taken as the edge weight.

        Arguments
        ---------
        weights: :class:`torch:torch.Tensor`
            The adjacency matrix of size ``(n, n)``.

        Returns
        -------
        :class:`torch:torch.Tensor`
            An ``(n, n)`` matrix adjacency matrix of the minimum spanning tree.

            Indices corresponding to the edges in the MST are set to one, rest
            are set to zero.

            If both weights[i, j] and weights[j, i] are non-zero, then the one
            will be located in whichever holds the *smaller* value (ties broken
            arbitrarily).
        """
        mst_matrix = mst(weights.cpu().numpy()).toarray() > 0
        # assert int(mst_matrix.sum()) + 1 == weights.size(0)
        return torch.Tensor(mst_matrix.astype(float))


class KSmallest(Function):
    """Return an indicator vector holing the smallest k elements in each row."""

    @staticmethod
    def forward(ctx, k, matrix):
        """Compute the positions holding the largest k elements in each row.

        Arguments
        ---------
        k: int
            How many elements to keep per row.
        matrix: :class:`torch:torch.Tensor`
            Tensor of size (n, m)

        Returns
        -------
        torch.Tensor of size (n, m)
           The positions that correspond to the k largest elements are set to
           one, the rest are set to zero."""
        ctx.mark_non_differentiable(matrix)
        matrix = matrix.numpy()
        indices = np.argsort(matrix, axis=1)
        mins = np.zeros_like(matrix)
        rows = np.arange(matrix.shape[0]).reshape(-1, 1)
        mins[rows, indices[:, :k]] = 1
        return torch.Tensor(mins)

class KLargest(Function):
    """Return an indicator vector holing the smallest k elements in each row."""

    @staticmethod
    def forward(ctx, k, matrix):
        """Compute the positions holding the largest k elements in each row.

        Arguments
        ---------
        k: int
            How many elements to keep per row.
        matrix: :class:`torch:torch.Tensor`
            Tensor of size (n, m)

        Returns
        -------
        torch.Tensor of size (n, m)
           The positions that correspond to the k largest elements are set to
           one, the rest are set to zero."""
        ctx.mark_non_differentiable(matrix)
        matrix = matrix.numpy()
        indices = np.argsort(-matrix, axis=1)
        mins = np.zeros_like(matrix)
        rows = np.arange(matrix.shape[0]).reshape(-1, 1)
        mins[rows, indices[:, :k]] = 1
        return torch.Tensor(mins)


class FRStatistic(object):
    """The classical Friedman-Rafsky test :cite:`friedman1979multivariate`.

    Arguments
    ----------
    n_1: int
        The number of data points in the first sample.
    n_2: int
        The number of data points in the second sample."""
    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

    def __call__(self, sample_1, sample_2, norm=2, ret_matrix=False):
        """Evaluate the non-smoothed Friedman-Rafsky test statistic.

        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, variable of size ``(n_1, d)``.
        sample_2: :class:`torch:torch.autograd.Variable`
            The second sample, variable of size ``(n_1, d)``.
        norm: float
            Which norm to use when computing distances.
        ret_matrix: bool
            If set, the call with also return a second variable.

            This variable can be then used to compute a p-value using
            :py:meth:`~.FRStatistic.pval`.

        Returns
        -------
        float
            The number of edges that do connect points from the *same* sample.
        """
        n_1 = sample_1.size(0)
        assert n_1 == self.n_1 and sample_2.size(0) == self.n_2
        sample_12 = torch.cat((sample_1, sample_2), 0)
        diffs = pdist(sample_12, sample_12, norm=norm)
        mst_matrix = MSTFn.apply(diffs)

        statistic = mst_matrix[:n_1, :n_1].sum() + mst_matrix[n_1:, n_1:].sum()

        if ret_matrix:
            return statistic, mst_matrix
        else:
            return statistic

    def pval(self, mst, n_permutations=1000):
        r"""Compute a p-value using a permutation test.

        Arguments
        ---------
        matrix: :class:`torch:torch.autograd.Variable`
            The matrix computed using :py:meth:`~.FRStatistic.__call__`.
        n_permutations: int
            The number of random draws from the permutation null.

        Returns
        -------
        float
            The estimated p-value."""
        return permutation_test_mat(mst.data.numpy(),
                                    self.n_1, self.n_2, n_permutations)


class KNNStatistic(object):
    """The classical k-NN test :cite:`friedman1983graph`.

    Arguments
    ---------
    n_1: int
        The number of data points in the first sample.
    n_2: int
        The number of data points in the second sample
    k: int
        The number of nearest neighbours (k in kNN).
    """
    def __init__(self, n_1, n_2, k):
        self.n_1 = n_1
        self.n_2 = n_2
        self.k = k

    def __call__(self, sample_1, sample_2, norm=2, ret_matrix=False):
        """Evaluate the non-smoothed kNN test statistic.

        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, variable of size ``(n_1, d)``.
        sample_2: :class:`torch:torch.autograd.Variable`
            The second sample, variable of size ``(n_1, d)``.
        norm: float
            Which norm to use when computing distances.
        ret_matrix: bool
            If set, the call with also return a second variable.

            This variable can be then used to compute a p-value using
            :py:meth:`~.KNNStatistic.pval`.

        Returns
        -------
        :class:`float`
            The number of edges that connect points from the *same* sample.
        :class:`torch:torch.autograd.Variable` (optional)
            Returned only if ``ret_matrix`` was set to true."""
        n_1 = sample_1.size(0)
        n_2 = sample_2.size(0)
        assert n_1 == self.n_1 and n_2 == self.n_2
        n = self.n_1 + self.n_2
        sample_12 = torch.cat((sample_1, sample_2), 0)
        diffs = pdist(sample_12, sample_12, norm=norm)
        indices = (1. - torch.eye(n)).byte()
        if sample_12.is_cuda:
            indices = indices.cuda()

        for i in range(n):
            diffs[i, i] = float('inf')  # We don't want the diagonal selected.

        smallest = KSmallest.apply(self.k, diffs.cpu())
        statistic = smallest[:n_1, :n_1].sum() + smallest[n_1:, n_1:].sum()

        print("KNN:",statistic)
        if ret_matrix:
            return statistic, smallest
        else:
            return statistic

    def pval(self, margs, n_permutations=1000):
        r"""Compute a p-value using a permutation test.

        Arguments
        ---------
        matrix: :class:`torch:torch.autograd.Variable`
            The matrix computed using :py:meth:`~.KNNStatistic.__call__`.
        n_permutations: int
            The number of random draws from the permutation null.

        Returns
        -------
        float
            The estimated p-value."""
        return permutation_test_mat(margs.data.cpu().numpy(),
                                    self.n_1, self.n_2, n_permutations)


class SmoothKNNStatistic(object):
    """The classical k-NN test :cite:`friedman1983graph`.

    Arguments
    ---------
    n_1: int
        The number of data points in the first sample.
    n_2: int
        The number of data points in the second sample
    k: int
        The number of nearest neighbours (k in kNN).
    """
    def __init__(self, n_1, n_2, alpha, k):
        self.n_1 = n_1
        self.n_2 = n_2
        self.alpha = alpha
        self.k = k

    def __call__(self, sample_1, sample_2, norm=2, ret_matrix=False):
        """Evaluate the non-smoothed kNN test statistic.

        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, variable of size ``(n_1, d)``.
        sample_2: :class:`torch:torch.autograd.Variable`
            The second sample, variable of size ``(n_1, d)``.
        norm: float
            Which norm to use when computing distances.
        ret_matrix: bool
            If set, the call with also return a second variable.

            This variable can be then used to compute a p-value using
            :py:meth:`~.KNNStatistic.pval`.

        Returns
        -------
        :class:`float`
            The number of edges that connect points from the *same* sample.
        :class:`torch:torch.autograd.Variable` (optional)
            Returned only if ``ret_matrix`` was set to true."""
        n_1 = sample_1.size(0)
        n_2 = sample_2.size(0)
        assert n_1 == self.n_1 and n_2 == self.n_2
        n = self.n_1 + self.n_2
        sample_12 = torch.cat((sample_1, sample_2), 0)
        diffs = pdist(sample_12, sample_12, norm=norm)

        for i in range(n):
            diffs[i, i] = float('inf')  # We don't want the diagonal selected.
        diff = diffs.numpy()
        sortind = np.array([np.argsort(l)[:(self.k)] for l in diff])
        dsort = np.array([l[np.argsort(l)[:(self.k)]] for l in diff])
        NN = sortind[:, :(self.k)]
        NN_T = NN.T

        Ak = np.zeros((self.k, n, n))
        for kind in range(self.k):

            for ind in range(n):
                NN_T_kindind = NN_T[kind, ind]
                Ak[kind, ind, NN_T_kindind] = 1
        bjk = np.zeros((n, self.k))

        for snd in range(self.k):
            bjk[:, snd] = np.sum(Ak[snd, :], axis=0)

        T12 = 0
        T22 = 0
        for snd in range(self.k):
            a = np.array([k for k in range(self.k)])
            b = np.array([snd])
            setdiff = np.setdiff1d(a, b)
            for rnd in setdiff:
                T12 = T12 + np.sum(np.multiply(Ak[snd, :, :], Ak[rnd, :, :].T))
                T22 = T22 + np.sum((np.multiply(bjk[:, snd], bjk[:, rnd])).T)

        AK_un = torch.tensor(Ak)
        Akt = AK_un.permute(0, 2, 1)

        T11 = np.sum(np.array(Ak) * np.array(Akt))
        T21 = np.sum(np.array(bjk) * np.array(bjk))
        T23 = np.sum(np.array(bjk))

        q = 4 * (n_1 - 1) * (n_2 - 1) / ((n - 2) * (n - 3))
        EL = self.k * (n_1 * (n_1 - 1) + n_2 * (n_2 - 1)) / (n - 1)

        varLg2 = ((n_1 * n_2) / (n - 1)) * (
                    q * (self.k - (2 * self.k * self.k) / (n - 1) + T11 / n + T12 / n) + (1 - q) * (
                        T21 / n + (2 * T22) / n - (2 * self.k * T23) / n + self.k * self.k))

        kount =np.sum(np.where(NN[:(n_1), :]<=(n_1-1),1,0))+np.sum(np.where(NN[(n_1):, :]>(n_1-1),1,0))
        t_val = (kount - EL)/math.sqrt(varLg2)
        print("SmoothKNN:",kount)
        return t_val,kount

    def pval(self, t_val):
        r"""Compute a p-value using a permutation test.

        Arguments
        ---------
        matrix: :class:`torch:torch.autograd.Variable`
            The matrix computed using :py:meth:`~.KNNStatistic.__call__`.
        n_permutations: int
            The number of random draws from the permutation null.

        Returns
        -------
        float
            The estimated p-value."""
        return 1-scipy.stats.norm(0,1).cdf(t_val)


class KFNStatistic(object):
    """The classical k-NN test :cite:`friedman1983graph`.

    Arguments
    ---------
    n_1: int
        The number of data points in the first sample.
    n_2: int
        The number of data points in the second sample
    k: int
        The number of nearest neighbours (k in kNN).
    """

    def __init__(self, n_1, n_2, k):
        self.n_1 = n_1
        self.n_2 = n_2
        self.k = k

    def __call__(self, sample_1, sample_2, norm=2, ret_matrix=False):
        """Evaluate the non-smoothed kNN test statistic.

        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, variable of size ``(n_1, d)``.
        sample_2: :class:`torch:torch.autograd.Variable`
            The second sample, variable of size ``(n_1, d)``.
        norm: float
            Which norm to use when computing distances.
        ret_matrix: bool
            If set, the call with also return a second variable.

            This variable can be then used to compute a p-value using
            :py:meth:`~.KNNStatistic.pval`.

        Returns
        -------
        :class:`float`
            The number of edges that connect points from the *same* sample.
        :class:`torch:torch.autograd.Variable` (optional)
            Returned only if ``ret_matrix`` was set to true."""
        n_1 = sample_1.size(0)
        n_2 = sample_2.size(0)
        assert n_1 == self.n_1 and n_2 == self.n_2
        n = self.n_1 + self.n_2
        sample_12 = torch.cat((sample_1, sample_2), 0)
        # print(sample_12)
        diffs = pdist_within_class_one_dimension(sample_12, sample_12, norm=norm)
        diffswhere = np.zeros_like(diffs)

        # print(diffs)
        # # 1.
        # file = open(r"D:\wbb\shift\failing-loudly\failing-loudly-master\failing-loudly-master\failing-loudly-master\paper_results\multiv\diffs_before.csv", 'w')
        # # 2.
        # writer = csv.writer(file)
        # # 3.
        # data = diffs
        # writer.writerow(data)
        # # 4.
        # file.close()
        # print(sample_1.size)

        indices = (1. - torch.eye(n)).byte()
        if sample_12.is_cuda:
            indices = indices.cuda()

        for i in range(n):
            diffs[i, i] = float('0')  # We don't want the diagonal selected.
        # diffs[(diffs > 1.2275)] = 0

        # print(diffs)

        # # 1.
        # file = open(r"D:\wbb\shift\failing-loudly\failing-loudly-master\failing-loudly-master\failing-loudly-master\paper_results\multiv\diffs.csv", 'w')
        # # 2.
        # writer = csv.writer(file)
        # # 3.
        # data = diffs
        # writer.writerow(data)
        # # 4.
        # file.close()

        # diffswhere[(diffs>1.414)]=2
        # c = np.array(np.where(diffswhere>1))

        Largest = KLargest.apply(self.k, diffs.cpu())
        # for i in range(len(c[0])):
        #     Largest[c[0][i],c[1][i]]=0

        # # 1.
        # file = open(r"D:\wbb\shift\failing-loudly\failing-loudly-master\failing-loudly-master\failing-loudly-master\paper_results\multiv\Largest.csv", 'w')
        # # 2.
        # writer = csv.writer(file)
        # # 3.
        # data = Largest
        # writer.writerow(data)
        # # 4.
        # file.close()

        # statistic = Largest[:n_1, n_1:].sum()
        statistic = Largest[:n_1,n_1:].sum()
        statistic_left_bottom = Largest[n_1:,:n_1].sum()
        statistic_left_upper = Largest[:n_1, :n_1].sum()
        statistic_right_upper = Largest[:n_1,n_1:].sum()
        statistic_right_bottom = Largest[n_1:,n_1:].sum()
        delta =  1-(n_1/(n_2-1)*statistic_right_bottom-statistic_left_bottom)/(statistic_right_upper - n_2/(n_1-1)*statistic_left_upper)

        # print("statistic_left_bottom:", statistic_left_bottom)
        # print("statistic_right_bottom:", statistic_right_bottom)
        # print("statistic_right_upper:", statistic_right_upper)
        # print("statistic_left_upper:", statistic_left_upper)
        #
        #
        #
        # print("delta:",delta)
        print("KFN:",statistic)
        if ret_matrix:
            # return statistic,Largest,delta
            return statistic, Largest
        else:
            return statistic

    def pval(self, margs, n_permutations=1000):
        r"""Compute a p-value using a permutation test.

        Arguments
        ---------
        matrix: :class:`torch:torch.autograd.Variable`
            The matrix computed using :py:meth:`~.KNNStatistic.__call__`.
        n_permutations: int
            The number of random draws from the permutation null.

        Returns
        -------
        float
            The estimated p-value."""
        return permutation_test_mat_KFN(margs.data.cpu().numpy(),
                                    self.n_1, self.n_2, n_permutations)

class halfKFNStatistic(object):
    """The classical k-NN test :cite:`friedman1983graph`.

    Arguments
    ---------
    n_1: int
        The number of data points in the first sample.
    n_2: int
        The number of data points in the second sample
    k: int
        The number of nearest neighbours (k in kNN).
    """

    def __init__(self, n_1, n_2, k):
        self.n_1 = n_1
        self.n_2 = n_2
        self.k = k

    def __call__(self, sample_1, sample_2, norm=2, ret_matrix=False):
        """Evaluate the non-smoothed kNN test statistic.

        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, variable of size ``(n_1, d)``.
        sample_2: :class:`torch:torch.autograd.Variable`
            The second sample, variable of size ``(n_1, d)``.
        norm: float
            Which norm to use when computing distances.
        ret_matrix: bool
            If set, the call with also return a second variable.

            This variable can be then used to compute a p-value using
            :py:meth:`~.KNNStatistic.pval`.

        Returns
        -------
        :class:`float`
            The number of edges that connect points from the *same* sample.
        :class:`torch:torch.autograd.Variable` (optional)
            Returned only if ``ret_matrix`` was set to true."""
        n_1 = sample_1.size(0)
        n_2 = sample_2.size(0)
        assert n_1 == self.n_1 and n_2 == self.n_2
        n = self.n_1 + self.n_2
        sample_12 = torch.cat((sample_1, sample_2), 0)
        diffs = pdist_within_class(sample_12, sample_12, norm=norm)
        diffswhere = np.zeros_like(diffs)
        indices = (1. - torch.eye(n)).byte()
        if sample_12.is_cuda:
            indices = indices.cuda()
        for i in range(n):
            # for j in range(n):
            #     diffs[i, j] = - diffs[i, j]
            diffs[i, i] = float('-inf')  # We don't want the diagonal selected.
        Largest = KLargest.apply(self.k, diffs.cpu())

        # statistic = Largest[:n_1, n_1:].sum()/(n_1*self.k)
        statistic = Largest[:n_1,n_1:].sum()
        print("halfKFN:",statistic)


        if ret_matrix:
            return statistic,Largest
        else:
            return statistic

    def pval(self, margs, n_permutations=1000):
        r"""Compute a p-value using a permutation test.

        Arguments
        ---------
        matrix: :class:`torch:torch.autograd.Variable`
            The matrix computed using :py:meth:`~.KNNStatistic.__call__`.
        n_permutations: int
            The number of random draws from the permutation null.

        Returns
        -------
        float
            The estimated p-value."""
        return permutation_test_mat_halfKFN(margs.data.cpu().numpy(),
                                    self.n_1, self.n_2, n_permutations)


class halfKNNStatistic(object):
    """The classical k-NN test :cite:`friedman1983graph`.

    Arguments
    ---------
    n_1: int
        The number of data points in the first sample.
    n_2: int
        The number of data points in the second sample
    k: int
        The number of nearest neighbours (k in kNN).
    """
    def __init__(self, n_1, n_2, k):
        self.n_1 = n_1
        self.n_2 = n_2
        self.k = k

    def __call__(self, sample_1, sample_2, norm=2, ret_matrix=False):
        """Evaluate the non-smoothed kNN test statistic.

        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, variable of size ``(n_1, d)``.
        sample_2: :class:`torch:torch.autograd.Variable`
            The second sample, variable of size ``(n_1, d)``.
        norm: float
            Which norm to use when computing distances.
        ret_matrix: bool
            If set, the call with also return a second variable.

            This variable can be then used to compute a p-value using
            :py:meth:`~.KNNStatistic.pval`.

        Returns
        -------
        :class:`float`
            The number of edges that connect points from the *same* sample.
        :class:`torch:torch.autograd.Variable` (optional)
            Returned only if ``ret_matrix`` was set to true."""
        n_1 = sample_1.size(0)
        n_2 = sample_2.size(0)
        assert n_1 == self.n_1 and n_2 == self.n_2
        n = self.n_1 + self.n_2
        sample_12 = torch.cat((sample_1, sample_2), 0)
        diffs = pdist(sample_12, sample_12, norm=norm)

        indices = (1. - torch.eye(n)).byte()
        if sample_12.is_cuda:
            indices = indices.cuda()

        for i in range(n):
            diffs[i, i] = float('inf')  # We don't want the diagonal selected.

        smallest = KSmallest.apply(self.k, diffs.cpu())
        statistic = smallest[:n_1, :n_1].sum()
        statistic_left_bottom = smallest[n_1:, :n_1].sum()
        statistic_left_upper = smallest[:n_1, :n_1].sum()
        statistic_right_upper = smallest[:n_1, n_1:].sum()
        statistic_right_bottom = smallest[n_1:, n_1:].sum()
        # delta = 1 - (n_1 / (n_2 - 1) * statistic_right_bottom - statistic_left_bottom) / (
        #             statistic_right_upper - n_2 / (n_1 - 1) * statistic_left_upper)

        print("statistic_left_upper:", statistic_left_upper)
        print("statistic_left_bottom:", statistic_left_bottom)
        print("statistic_right_upper:", statistic_right_upper)
        print("statistic_right_bottom:", statistic_right_bottom)

        alpha1 = 1-statistic_right_upper/statistic_left_upper
        print(alpha1)
        print("halfKNN:",statistic)
        if ret_matrix:
            return alpha1, smallest
        else:
            return alpha1

    def pval(self, margs, n_permutations=1000):
        r"""Compute a p-value using a permutation test.

        Arguments
        ---------
        matrix: :class:`torch:torch.autograd.Variable`
            The matrix computed using :py:meth:`~.KNNStatistic.__call__`.
        n_permutations: int
            The number of random draws from the permutation null.

        Returns
        -------
        float
            The estimated p-value."""
        return permutation_test_mat(margs.data.cpu().numpy(),
                                    self.n_1, self.n_2, n_permutations)


class oursKFNStatistic(object):
    """The classical k-NN test :cite:`friedman1983graph`.

    Arguments
    ---------
    n_1: int
        The number of data points in the first sample.
    n_2: int
        The number of data points in the second sample
    k: int
        The number of nearest neighbours (k in kNN).
    """

    def __init__(self, n_1, n_2, alpha, k):
        self.n_1 = n_1
        self.n_2 = n_2
        self.alpha = alpha
        self.k = k

    def __call__(self, sample_1, sample_2, norm=2, ret_matrix=False):
        """Evaluate the non-smoothed kNN test statistic.

        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, variable of size ``(n_1, d)``.
        sample_2: :class:`torch:torch.autograd.Variable`
            The second sample, variable of size ``(n_1, d)``.
        norm: float
            Which norm to use when computing distances.
        ret_matrix: bool
            If set, the call with also return a second variable.

            This variable can be then used to compute a p-value using
            :py:meth:`~.KNNStatistic.pval`.

        Returns
        -------
        :class:`float`
            The number of edges that connect points from the *same* sample.
        :class:`torch:torch.autograd.Variable` (optional)
            Returned only if ``ret_matrix`` was set to true."""
        n_1 = sample_1.size(0)
        n_2 = sample_2.size(0)
        assert n_1 == self.n_1 and n_2 == self.n_2
        n = self.n_1 + self.n_2
        sample_12 = torch.cat((sample_1, sample_2), 0)
        diffs = pdist_within_class_one_dimension(sample_12, sample_12, norm=norm)
        diffswhere = np.zeros_like(diffs)
        indices = (1. - torch.eye(n)).byte()
        if sample_12.is_cuda:
            indices = indices.cuda()
        for i in range(n):
            # for j in range(n):
            #     diffs[i, j] = - diffs[i, j]
            diffs[i, i] = float('-inf')  # We don't want the diagonal selected.
        Largest = KLargest.apply(self.k, diffs.cpu())
        statistic = Largest[:n_1, n_1:].sum()/(n_1*self.k)

        return statistic,statistic

    def pval(self, t_val):
        r"""Compute a p-value using a permutation test.

        Arguments
        ---------
        matrix: :class:`torch:torch.autograd.Variable`
            The matrix computed using :py:meth:`~.KNNStatistic.__call__`.
        n_permutations: int
            The number of random draws from the permutation null.

        Returns
        -------
        float
            The estimated p-value."""
        return 1-scipy.stats.norm(0,1).cdf(t_val)