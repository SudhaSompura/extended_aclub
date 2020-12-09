import numpy as np
from sklearn.linear_model import LogisticRegression
import sys
import math
import random
import warnings
import true_ws as tw


def den_func(cj, w):
    val_dot = np.dot(cj, w)
    val1 = np.exp(val_dot)
    val2 = np.exp(-val_dot)
    return 1 / (2 + val1 + val2)


def logistic_func(v):
    if v < 0:
        return 1 - 1 / (1 + np.exp(v))
    else:
        return 1 / (1 + np.exp(-v))


def asmt_action_select(cj, w_estm, asmts, t1, t2, a_prev, Q, A, N_a, lam0, K,
                       s_mi, alpha=0.4):
    """
    Selects the (assessment,action) pair for a single student
    :param cj: Concept vector of student j
    :param w_estm: Estimated_ws
    :param asmts: List of assessments
    :param t1: Lower threshold for ZPD
    :param t2: Higher threshold for ZPD
    :param a_prev: Previous action suggested for the student
    :param Q: Number of questions per assessment
    :param A: Total number of actions
    :param N_a: Students suggested pairs over iterations
    :param lam0: Parameters
    :param K: Dimension of concept vector
    :param s_mi: score per question in assessments
    :param alpha: Parameter
    :return: (assessment, action) pair for the student
    """

    M = len(asmts)

    #  ZPD selection
    zpd = []
    for p in range(M):
        temp_list = [logistic_func(np.dot(cj, w_estm[a_prev, p, i])) for i in range(Q)]
        vp = (sum(temp_list)) / Q
        if t1 < vp < t2:
            zpd.append(p)

    # (asmt, action) selection
    mj = -1
    aj = -1
    F = np.empty((A, K, K))
    for a in range(A):
        m = np.random.choice(M)
        q = np.random.choice(Q)
        f_a = lam0 * np.eye(K) + sum([(np.outer(cj, cj) / (den_func(cj, w_estm[a, m, q])
                                                           + 0.0000000001))
                                      for i in range(N_a[(m, a)][0])])
        F[a] = f_a

    prev_sum = -sys.maxsize
    for a in range(A):
        f_inv = np.linalg.inv(F[a])
        for m in zpd:
            sum_val = 0
            for i in range(Q):
                np_dot_temp = np.dot(cj, w_estm[a, m, i])
                temp_val1 = f_inv.dot(cj)
                temp_val2 = np.dot(cj, temp_val1)
                temp_val3 = temp_val2 * alpha
                temp_val4 = temp_val3 / N_a[(m, a)][0]
                try:
                    temp2 = math.sqrt(temp_val4)
                except:
                    temp2 = 0

                c_dot_w = np_dot_temp + temp2
                sum_val = sum_val + (s_mi[i] * logistic_func(c_dot_w))

            if sum_val >= prev_sum:
                aj = a
                mj = m
                prev_sum = sum_val

    if mj == -1 or aj == -1:
        # print("picking randomly")
        mj = random.choice(range(M))
        aj = random.choice(range(A))

    return mj, aj


def estm_ws_am(prev_w_a, dataset, y):
    """
    Estimates the w_hat for a single w pair
    :param prev_w_a: Previous estimated w
    :param dataset: List of all the previous concept vectors
    :param y: Corresponding labels for the concept vectors
    :return: New Estimated w
    """
    try:
        c = sys.maxsize * 100000.0
        model = LogisticRegression(solver='liblinear', C=c, random_state=0)
        model.fit(dataset, y)
        w_ij_hat = model.coef_
    except:
        w_ij_hat = prev_w_a

    return w_ij_hat[0]


def graded_response(asmt, action, true_ws, cj, Q):
    """
    Returns the score for an (asmt,action) pair
    :param asmt: Assessment
    :param action: Action
    :param true_ws: True_w vectors
    :param cj: Concept vector of the student
    :param Q: Number of questions per assessment
    :return: Score of the student for the (asmt, action) pair
    """
    final_score = 0
    for q in range(Q):
        v = np.dot(cj, true_ws[action, asmt, q])
        inv_logit = logistic_func(v)
        score = np.random.binomial(1, inv_logit)
        final_score += score
    return final_score


def get_response(true_w, cj):
    v = np.dot(cj, true_w)
    inv_logit = logistic_func(v)
    score = np.random.binomial(1, inv_logit)
    return score


def asmt_act_truew_select(A, M, Q, t1, t2, true_ws, cj):
    """
    Returns the suggested (asmt,action) pairs using true_ws
    :param A: Number of actions
    :param M: Dict of assessment with difficulty
    :param Q: Number of questions per assessment
    :param t1: Lower threshold for ZPD
    :param t2: Higher threshold for ZPD
    :param true_ws: true_w values
    :param cj: Concept vector of the student
    :return: (asmt, action) pair for the student
    """
    results = []
    for a in range(A):
        for m in range(M):
            score = graded_response(m, a, true_ws, cj, Q)
            score = score / Q
            if t1 < score < t2:
                results.append((m, a))

    try:
        r = random.choice(results)
    except IndexError:
        print('picking randomly')
        r = (random.choice(range(M)), random.choice(range(A)))
    return r


def extended_aclub(C, A, M, Q, K, N_a, true_ws, a_prev, std_num, std_grades,
                   estm_ws=[], is_true_w=False):
    """
    Returns the next (assessment, action) pairs for all students
    :param C: List of concept vectors
    :param A: Number of actions
    :param M: List of assessments along with difficulty level
    :param Q: Number of questions per assessment
    :param K: Dimension of concept vectors
    :param N_a: Student suggested pairs over iterations
    :param true_ws: True_ws
    :param a_prev: Dict of previously suggested actions for each student
    :param std_num: Number of students
    :param std_grades: Students grades over iterations
    :param estm_ws: Estimated_ws
    :param is_true_w: Flag for whether results are for true_w or estm_w
    :return: (asmt, action) pair for each student
    """

    M_len = len(M)

    # set the thresholds for the ZPD
    t1 = 0.3
    t2 = 0.8
    result = np.empty((std_num, 3), dtype=int)

    # results for estimated_w values
    if not is_true_w:
        # lam0 constant
        lam0 = 0.5

        # s_mi vectors, (question, marks) pair of dictionary
        s_mi = {}
        for q in range(Q):
            s_mi[q] = 5

        # alpha constant
        alpha = 0.6

        for a in range(A):
            for m in range(M_len):
                std_list = N_a[(m, a)][1]
                for q in range(Q):
                    prev_w_a = estm_ws[a, m, q]
                    dataset = []
                    y = []
                    for std in std_list:
                        dataset.extend(std_grades[(m, a, std)][q][1])
                        y.extend(std_grades[(m, a, std)][q][0])
                    estm_ws[a, m, q] = estm_ws_am(prev_w_a, dataset, y)

        for j in range(std_num):
            mj, aj = asmt_action_select(C[j], estm_ws, M, t1, t2, a_prev[j], Q, A, N_a,
                                        lam0, K, s_mi, alpha)
            a_prev[j] = aj
            N_a_update = N_a.get((mj, aj), [1, [np.random.choice(std_num), ]])
            actn_select = N_a_update[0] + 1  # increase the number of times aj is prescribed
            stds = list(set(N_a_update[1] + [j, ]))
            N_a[(mj, aj)] = [actn_select, stds]
            result[j][0] = j
            result[j][1] = mj
            result[j][2] = aj

            # update std_grades
            for q in range(Q):
                std_grades[(mj, aj, j)][q][0].append(get_response(true_ws[aj, mj, q],
                                                                  C[j]))
                std_grades[(mj, aj, j)][q][1].append(C[j])

    # getting the results for the true_w values
    else:
        for j in range(std_num):
            mj, aj = asmt_act_truew_select(A, M_len, Q, t1, t2, true_ws, C[j])
            result[j][0] = j
            result[j][1] = mj
            result[j][2] = aj

    return result


def expected_score(asmt, action, true_ws, cj, Q):
    score = 0
    for q in range(Q):
        v = np.dot(cj, true_ws[action, asmt, q])
        inv_logit = logistic_func(v)
        score += inv_logit
    return score


def get_graded_response(std_num, num_iters):
    """
    Returns the next assessment & action pair for each student over num_iters number
    of iterations
    :param std_num: Number of students
    :param num_iters: Number of iterations to run
    :return: (asmt,action) pair for each student over num_iter iterations
    """

    # warnings.simplefilter('error', RuntimeWarning)

    A = 1
    # M = {0:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2}
    M = {0: 0, 1: 0}
    Q = 10
    K = 2
    num_categories = 1
    true_ws = tw.get_true_ws(A, M, Q, K, num_categories)

    # set the N_a dictionary (action, n_a) pairs
    C = np.random.multivariate_normal(np.zeros(K), np.eye(K), size=std_num)
    M_len = len(M)
    N_a = {}
    for m in range(M_len):
        for a in range(A):
            N_a[(m, a)] = [1, [np.random.choice(std_num), ]]

    std_grades = {}
    for m in range(M_len):
        for a in range(A):
            for s in range(std_num):
                std_grades[(m, a, s)] = {}
                for q in range(Q):
                    if s in N_a[(m, a)][1]:
                        g = get_response(true_ws[a, m, q], C[s])
                        std_grades[(m, a, s)][q] = [[g, ], [C[s], ]]
                    else:
                        std_grades[(m, a, s)][q] = [[], []]

    estm_ws = np.random.multivariate_normal(-2 * np.ones(K), np.eye(K), size=(A, M_len, Q))
    std_scores_truew = np.empty((std_num, num_iters))
    std_scores_estmw = np.empty((std_num, num_iters))
    suggested_pairs = np.empty((std_num, num_iters, 2))

    norm_diff = np.empty((A, M_len, Q, num_iters))
    a_prev = {}
    for j in range(std_num):
        a_prev[j] = random.choice(range(A))

    for i in range(num_iters):
        C = np.random.multivariate_normal(np.zeros(K), np.eye(K), size=std_num)
        result = extended_aclub(C, A, M, Q, K, N_a, true_ws, a_prev, std_num,
                                std_grades, estm_ws=estm_ws)

        for a in range(A):
            for m in range(M_len):
                for q in range(Q):
                    diff = true_ws[a, m, q] - estm_ws[a, m, q]
                    norm_diff[a, m, q, i] = np.linalg.norm(diff)

        for r in result:
            s_num = r[0]
            std_scores_estmw[s_num, i] = expected_score(r[1], r[2], estm_ws, C[s_num], Q)
            std_scores_truew[s_num, i] = expected_score(r[1], r[2], true_ws, C[s_num], Q)
            suggested_pairs[s_num, i, 0] = r[1]
            suggested_pairs[s_num, i, 1] = r[2]

    return std_scores_truew, std_scores_estmw, suggested_pairs, norm_diff

