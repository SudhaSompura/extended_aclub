import matplotlib.pyplot as plt
import asmt_action_sugg


if __name__ == '__main__':
    std_num = 5
    num_iters = 500
    std_scores_truew, std_scores_estmw, suggested_pairs, norm_diff = \
        asmt_action_sugg.get_graded_response(std_num, num_iters)

    a, m, q, temp = norm_diff.shape
    for i in range(a):
        for j in range(m):
            for k in range(q):
                plt.subplots(figsize=(9, 5))
                plt.plot(norm_diff[i, j, k], linewidth=2)
                plt.xlabel('iteration')
                plt.ylabel('L2 norm difference')
                # plt.title('true_w - estm_w for (A,M,Q) pair ({} {} {})'.format(0, 1, 1))
                plt.tight_layout()
                plt.grid()
                # plt.savefig("L2_norm_1.jpg", dpi=150)
                plt.show()

    for std in range(std_num):
        plt.subplots(figsize=(9, 5))
        plt.plot(std_scores_truew[std] - std_scores_estmw[std])
        plt.xlabel('iteration')
        plt.ylabel('Expected score diff')
        # plt.title('Student expected score difference over iterations')
        plt.tight_layout()
        plt.grid()
        plt.savefig("score_diff_{}.png".format(std), dpi=150)
        plt.show()

