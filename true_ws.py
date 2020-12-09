import numpy as np


def act_asmt_true_w(mean, cov, Q, K, diff=0.1):
    """
    Returns the true w vecs for an (asmt, action) pair.
       Size: Q * K
    :param mean: mean of the easy category
    :param cov: covariance matrix for true_ws
    :param Q: Number of questions per assessment
    :param K: Dimension of the concept vector
    :param diff: Difference range in true ws  for questions within each assessment
    :return: Q*K array of true_ws
    """

    true_w = np.empty((Q, K))
    # sample K-vector from gaussian with given mean & cov for first question
    w = np.random.multivariate_normal(mean, cov)
    true_w[0] = w

    for q in range(Q - 1):
        d = np.random.uniform(low=-diff, high=diff, size=K)  # add a given random difference
        true_w[q + 1] = w + d

    return true_w


def action_true_w(a, M, Q, K, mu_easy):
    """
    Generates the true_ws for a given action for all the asmts
    :param a:
    :param M: Number of assessments
    :param Q: Number of questions per assessment
    :param K: Dimension of concept vector
    :param mu_easy: mean of the easy category for this action
    :return: true_ws for all the assessment for the action
    """
    num_asmt = len(M)
    actn_true_w = np.empty((num_asmt, Q, K))

    for m in M.items():
        mu_easy = mu_easy + 0.1

        ''' generate a mean according to asmt difficulty level
            easy ones get higher mean & difficult ones get lower mean coz
            inverse logit function '''

        mean = (mu_easy + m[1]) * np.ones(K)
        cov = np.eye(K)
        actn_true_w[m[0]] = act_asmt_true_w(mean, cov, Q, K)

    return actn_true_w


def get_true_ws(A, M, Q=10, K=5, num_categories=3):
    """
    Generates the true_w vectors for all (action, asmt, Q) pairs
     :param A: Total number of actions
     :param M: Total number of assessments
     :param Q: Number of questions per assessment
     :param K: Dimension of concept vector
     :param num_categories: Total number of difficulty categories in assessments
     :return: A*M*Q*K shape array of true_ws
     """

    num_asmt = len(M)
    true_w = np.empty((A, num_asmt, Q, K))
    mu_easy = 0.1

    for a in range(A):
        true_w[a] = action_true_w(a, M, Q, K, mu_easy)
        # changing the mean for easy category of action
        mu_easy = mu_easy + num_categories * 2

    return true_w

