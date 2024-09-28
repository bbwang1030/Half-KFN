import torch


def pdist(sample_1, sample_2, norm=2, eps=1e-60):
    r"""Compute the matrix of all squared pairwise distances.

    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.

    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    torch.set_printoptions(precision=60, sci_mode=True)
    if norm == 2.:
        # sample_1 = torch.log(sample_1)
        # sample_2 = torch.log(sample_2)
        sample_1 = torch.exp((torch.log(sample_1)) / 10)
        # print(sample_1)
        sample_2 = torch.exp((torch.log(sample_2)) / 10)
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)


def pdist_within_class(sample_1, sample_2, norm=2, eps=1e-60):
    r"""Compute the matrix of all squared pairwise distances.

    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.

    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""

    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    torch.set_printoptions(precision=60, sci_mode=True)
    if norm == 2.:
        sample_1 = torch.exp((torch.log(sample_1))/10)
        # print(sample_1)
        sample_2 = torch.exp((torch.log(sample_2))/10)
        # sample_1 = sample_1.to(dtype= torch.double)
        # sample_2 = sample_2.to(dtype=torch.double)

        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        # print("norms_1",norms_1)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        # print("norms",norms)
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        # print("distances_squared:",distances_squared)
        max_indices = sample_1.max(1).indices
        # print(sample_1)
        max_one_zeros = torch.zeros_like(sample_1)
        for i,indices in enumerate(max_indices):
            max_one_zeros[i, indices] = 1
        # print(max_one_zeros)
        Determine_belong_same_class = max_one_zeros.mm(max_one_zeros.t())
        # print(eps + torch.abs(distances_squared))
        diffs = torch.sqrt(eps + torch.abs(distances_squared))
        # print(diffs)
        diffs[(Determine_belong_same_class == 0)] = 0
        # print(diffs)
        return diffs
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

def pdist_within_class_one_dimension(sample_1, sample_2, norm=2, eps=1e-60):
    r"""Compute the matrix of all squared pairwise distances.

    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.

    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""

    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    torch.set_printoptions(precision=60, sci_mode=True)
    # print(sample_1)
    if norm == 2.:
        max_indices = sample_1.max(1).indices
        # print(max_indices)
        max_one_zeros = torch.zeros_like(sample_1)
        sample_one_dimension = torch.zeros([n_1, 1])
        for i, indices in enumerate(max_indices):
            # print(i, indices)
            sample_one_dimension[i, 0] = sample_1[i, indices]
            max_one_zeros[i, indices] = 1
        # print(sample_one_dimension)
        Determine_belong_same_class = max_one_zeros.mm(max_one_zeros.t())
        # print(Determine_belong_same_class)

        sample_1 = torch.log(sample_one_dimension)
        sample_2 = torch.log(sample_one_dimension)
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        # print("norms_1",norms_1)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        # print("norms",norms)
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        diffs = torch.sqrt(eps + torch.abs(distances_squared))
        diffs[(Determine_belong_same_class == 0)] = 0
        # print(diffs)
        return diffs
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)



def pdist_within_class_validation(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.

    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.

    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    n_1_half = int(n_1/2)
    nclass = sample_1.size(1)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        # print(norms_1.shape,norms_1)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        # print(norms.shape,norms)
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        # print(distances_squared.shape,distances_squared)
        max_indices = sample_1.max(1).indices
        # print(max_indices,n_1)
        max_one_zeros = torch.zeros_like(sample_1)
        for i,indices in enumerate(max_indices):
            max_one_zeros[i, indices] = 1
        # print(torch.nonzero(max_indices[:n_1_half] == 0))
        E = 0
        for c_ in range(nclass):
            n_c1 = (max_indices[:n_1_half] == c_).sum()
            n_c1_sample = sample_1[torch.nonzero(max_indices[:n_1_half] == c_)]
            n_c2 = (max_indices[n_1_half:] == c_).sum()
            n_c2_sample = sample_1[torch.nonzero(max_indices[n_1_half:] == c_)]
            # print(n_c1,n_c2,n_c1_sample)
            E_class = 2*n_c1*n_c1*n_c2*k/((n_c1+n_c2)*(n_c1+n_c2-1))
            E = E + E_class
        print("Expectation:",E)
        #     for i in max_one_zeros[]:
        #     torch.cat((sample_1, sample_2), 0)
        # print(max_one_zeros.shape)
        Determine_belong_same_class = max_one_zeros.mm(max_one_zeros.t())
        # print(Determine_belong_same_class)
        diffs = torch.sqrt(eps + torch.abs(distances_squared))
        diffs[(Determine_belong_same_class == 0)] = 0
        # print(diffs)
        return diffs
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)


def each_class_sample(sample_1, sample_2, norm=2, eps=1e-5, n1 = 1):
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    nclass = 1
    norm = float(norm)
    # print(sample_1)
    if norm == 2.:
        norms_1 = torch.sum(sample_1 ** 2, dim=1, keepdim=True)
        # print(norms_1.shape,norms_1)
        norms_2 = torch.sum(sample_2 ** 2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        # print(norms.shape,norms)
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        # print(distances_squared.shape,distances_squared)
        max_indices = sample_1.max(1).indices
        # print(max_indices,n_1)
        max_one_zeros = torch.zeros_like(sample_1)
        for i, indices in enumerate(max_indices):
            max_one_zeros[i, indices] = 1
        # print(torch.nonzero(max_indices[:n1] == 0))
        E = 0
        VAR = 0
        n_c1 = [0]
        n_c2 = [0]
        n_c1_sample_list = []
        n_c2_sample_list = []
        k=4
        print(nclass)
        for c_ in range(nclass):
            # print("n1",max_indices[:n1])
            n_c1[c_] = (max_indices[:n1] == c_).sum()
            n_c1_sample = sample_1[torch.nonzero(max_indices[:n1] == c_)].reshape([n_c1[c_],-1])
            # n_c1_sample = sample_1[torch.nonzero(max_indices[:n1] == c_)]
            n_c2[c_] = (max_indices[n1:] == c_).sum()
            # print(max_indices[n1:])
            n_c2_sample = sample_1[torch.nonzero(max_indices[n1:] == c_)+n1].reshape([n_c2[c_],-1])
            # n_c2_sample = sample_1[torch.nonzero(max_indices[n1:] == c_) + n1]
            # print(torch.nonzero(max_indices[n1:] == c_)+n1,n_c2_sample)
            n_c1_sample_list.append(n_c1_sample)
            n_c2_sample_list.append(n_c2_sample)
            print("n_c1[c_],n_c2[c_]",n_c1[c_],n_c2[c_])
            # E_class = 2 * n_c1[c_] * n_c1[c_] * n_c2[c_] * k / ((n_c1[c_] + n_c2[c_]) * (n_c1[c_] + n_c2[c_] - 1))
            E_class = k * n_c1[c_] * n_c2[c_] / (n_c1[c_] + n_c2[c_] - 1)
            VAR_class = k * n_c1[c_] * n_c2[c_] * (n_c1[c_] + n_c2[c_] - 1 - k) / ((n_c1[c_] + n_c2[c_] - 1) *(n_c1[c_] + n_c2[c_] - 1))
            # VAR_class = n_c1[c_] * n_c2[c_] * (n_c1[c_]-1) * (n_c2[c_] - 1) * (k/(n_c1[c_] + n_c2[c_] - 1))**2 + n_c1[c_] *  n_c2[c_] *(n_c2[c_] - 1) *k/(n_c1[c_] + n_c2[c_] - 1) *(k-1)/(n_c1[c_] + n_c2[c_] - 2) + n_c1[c_] * n_c2[c_] * (n_c1[c_]-1) * (k/(n_c1[c_] + n_c2[c_] - 1))**2 + n_c1[c_] * n_c2[c_]*k/(n_c1[c_] + n_c2[c_] - 1)
            print("E_class,VAR_class",E_class,VAR_class)
            E = E + E_class
            VAR = VAR +VAR_class
        print("Expectation,VAR:", E, VAR)
        # n_c1_sample = torch.cat(n_c1_sample_list,dim = 0)
        # n_c2_sample = torch.cat(n_c2_sample_list, dim=0)
        #     for i in max_one_zeros[]:
        #     torch.cat((sample_1, sample_2), 0)
        # print(max_one_zeros.shape)
        Determine_belong_same_class = max_one_zeros.mm(max_one_zeros.t())
        # print(Determine_belong_same_class)
        diffs = torch.sqrt(eps + torch.abs(distances_squared))
        diffs[(Determine_belong_same_class == 0)] = 0
        # print(diffs)
        return n_c1, n_c2, n_c1_sample_list, n_c2_sample_list
