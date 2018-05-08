import math
from math import log

import numpy as np


class RankingTest:
    ptype_id = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    prototypes_per_class = 2
    ranking_range = [0, 5]
    ranking_list = np.array(list(range(ranking_range[0], ranking_range[1] + 1)))
    gaussian_sd = 0.5
    omega_ = np.eye(2)
    W_ = np.array([[1.1, 9.5], [2.1, 9.8], [20, 20], [32, -1.7], [0.5, -2.5], [9.8, -3.4], [-9.4, -3.6]
                                , [-2.8, -9.2], [-3.1, -9.5], [-3.7, 9.9], [-9.9, 0.1], [-0.5, 0.5]])


    def _costfunc(self, data_point, label, k_size):
        w_plus, w_minus, max_error_cls = self.find_prototype(data_point, label, k_size)

        sum_cost = 0
        while len(w_plus) > 0 and len(w_minus) > 0:
            min_value = np.inf
            min_ind_correct = 0
            index = 0
            for prot in w_plus:
                if prot[1] < min_value:
                    min_value = prot[1]
                    min_ind_correct = index
                index += 1
            closest_cor_p = w_plus.pop(min_ind_correct)

            # find closest wrong prototype from w_minus
            min_value = np.inf
            min_ind_wrong = 0
            index = 0
            for prot in w_minus:
                if prot[1] < min_value:
                    min_value = prot[1]
                    min_ind_wrong = index
                index += 1
            closest_wro_p = w_minus.pop(min_ind_wrong)

            # update prototypes and omega here
            pt_pair = [closest_cor_p, closest_wro_p]
            alpha_distance_plus, alpha_plus = self.alpha_dist_plus(pt_pair, label)
            alpha_distance_minus, alpha_minus = self.alpha_dist_minus(pt_pair, label, max_error_cls)
            mu = (alpha_distance_plus - alpha_distance_minus) / (alpha_distance_plus + alpha_distance_minus)
            sum_cost += mu

        return sum_cost


    def find_prototype(self, data_point, label, k_size):
        list_dist = _squared_euclidean(self.W_.dot(self.omega_.conj().T), data_point.dot(self.omega_.conj().T)).flatten()
        print(list_dist)

        # print(list_dist, label, self.ptype_id)
        # correct kernel class
        correct_idx0 = label * self.prototypes_per_class
        correct_idx1 = correct_idx0 + self.prototypes_per_class
        # print(list_dist[correct_idx0:correct_idx1])

        self.ranking_list
        correct_cls_min = label - k_size
        correct_cls_max = label + k_size
        if correct_cls_min < self.ranking_range[0]:
            correct_cls_min = self.ranking_range[0]
        if correct_cls_max > self.ranking_range[1]:
            correct_cls_max = self.ranking_range[1]

        correct_ranking = np.array(list(range(correct_cls_min, correct_cls_max + 1)))
        # print("correct_ranking: ", correct_ranking)
        # print(self.ranking_list)

        # all classes with True and False
        class_list = np.zeros((len(self.ptype_id)//self.prototypes_per_class), dtype=bool)
        class_list[correct_ranking] = True
        # print(class_list)

        # collection set of closest prototype from each correct class
        # collection set of closest prototype from each wrong class
        cls_ind = 0
        W_plus = []
        W_minus = []
        max_error_cls = 0
        for correct_cls in class_list:
            ind0 = cls_ind * self.prototypes_per_class
            ind1 = ind0 + self.prototypes_per_class
            min_val = min(list_dist[ind0:ind1])
            min_idx = np.argmin(list_dist[ind0:ind1], axis=0) + ind0
            # print(min_idx, min_val)
            if correct_cls:
                W_plus.append([min_idx, min_val])
            else:
                W_minus.append([min_idx, min_val])
                if abs(cls_ind - label) > max_error_cls:
                    max_error_cls = abs(cls_ind - label)
            cls_ind += 1
        # print(W_plus, W_minus)
        return W_plus, W_minus, max_error_cls

    # update prototype a and b, and omega
    def update_prot_and_omega(self, w_plus, w_minus, label, max_error_cls, datapoint):
        # print(w_plus, w_minus)
        while len(w_plus) > 0 and len(w_minus) > 0:
            # find closest correct prototype from w_plus
            min_value = np.inf
            min_ind_correct = 0
            index = 0
            for prot in w_plus:
                if prot[1] < min_value:
                    min_value = prot[1]
                    min_ind_correct = index
                index += 1
            closest_cor_p = w_plus.pop(min_ind_correct)

            # find closest wrong prototype from w_minus
            min_value = np.inf
            min_ind_wrong = 0
            index = 0
            for prot in w_minus:
                if prot[1] < min_value:
                    min_value = prot[1]
                    min_ind_wrong = index
                index += 1
            closest_wro_p = w_minus.pop(min_ind_wrong)

            # update prototypes and omega here
            pt_pair = [closest_cor_p, closest_wro_p]
            delta_correct_prot, delta_wrong_prot, delta_omega = self._derivatives(pt_pair, label, max_error_cls, datapoint)

            pid_correct = closest_cor_p[0]
            pid_wrong = closest_wro_p[0]
            # print(self.W_)
            print(self.omega_)
            self.W_[pid_correct] = self.W_[pid_correct] - delta_correct_prot
            self.W_[pid_wrong] = self.W_[pid_wrong] - delta_wrong_prot
            self.omega_ = self.omega_ - delta_omega
            print(self.W_)
            print(self.omega_)



    # calculate derivatives of prototypes a, b and omega
    def _derivatives(self, pt_pair, label, max_error_cls, datapoint):
        # print(max_error_cls)
        # calculate alpha+ and alpha-
        alpha_distance_plus, alpha_plus = self.alpha_dist_plus(pt_pair, label)
        alpha_distance_minus, alpha_minus = self.alpha_dist_minus(pt_pair, label, max_error_cls)

        gamma_plus = 2*alpha_distance_minus / pow((alpha_distance_plus + alpha_distance_minus), 2)
        gamma_minus = -2*alpha_distance_plus / pow((alpha_distance_plus + alpha_distance_minus), 2)

        pid_correct = pt_pair[0][0]
        pid_wrong = pt_pair[1][0]
        diff_correct = datapoint - self.W_[pid_correct]
        diff_wrong = datapoint - self.W_[pid_wrong]

        diff_mtx_correct = diff_correct.T.dot(diff_correct)
        delta_omega_plus = gamma_plus * 2 * alpha_plus*self.omega_.dot(diff_mtx_correct)

        diff_mtx_wrong = diff_wrong.T.dot(diff_wrong)
        delta_omega_minus = gamma_minus * 2 * alpha_minus * self.omega_.dot(diff_mtx_wrong)

        delta_omega = delta_omega_plus + delta_omega_minus
        print("delta_omega:", delta_omega)

        delta_correct_prot = gamma_plus * (-2*alpha_plus*diff_correct.dot(self.omega_.T.dot(self.omega_)))
        delta_wrong_prot = gamma_minus * (-2*alpha_minus*diff_wrong.dot(self.omega_.T.dot(self.omega_)))
        # print("delta:", delta_correct_prot, delta_wrong_prot)

        return delta_correct_prot, delta_wrong_prot, delta_omega

    def alpha_dist_plus(self, pt_pair, label):
        distance_correct = pt_pair[0][1]
        ranking_diff_correct = abs(label - pt_pair[0][0] // self.prototypes_per_class)

        alpha_plus = math.exp(- pow(ranking_diff_correct, 2) / (2 * pow(self.gaussian_sd, 2)))

        alpha_distance_plus = alpha_plus * distance_correct

        return alpha_distance_plus, alpha_plus

    def alpha_dist_minus(self, pt_pair, label, max_error_cls):
        distance_wrong = pt_pair[1][1]
        ranking_diff_wrong = abs(label - pt_pair[1][0] // self.prototypes_per_class)

        alpha_minus = math.exp(- pow(max_error_cls - ranking_diff_wrong, 2) / (2 * pow(self.gaussian_sd, 2))) * \
                      math.exp(-pow(distance_wrong, 2) / (2 * pow((1 - self.gaussian_sd), 2)))

        alpha_distance_minus = alpha_minus * distance_wrong

        return alpha_distance_minus, alpha_minus


def _squared_euclidean(a, b=None):
    if b is None:
        d = np.sum(a ** 2, 1)[np.newaxis].T + np.sum(a ** 2, 1) - 2 * a.dot(
            a.T)
    else:
        d = np.sum(a ** 2, 1)[np.newaxis].T + np.sum(b ** 2, 1) - 2 * a.dot(
            b.T)
    return np.maximum(d, 0)


rankTest = RankingTest()
data_point = np.array([[0, 0]])

# l_distance = np.array([4.11044186, 6.31040921, 58.6941359,  53.96639758, 32.68570672, 39.96803298,
#                        11.14983586,  8.34313266, 18.74119665, 11.66475519, 28.03848346, 29.49774315])
lbl = 1
kernel_size = 1
init_cost = np.inf
cost = np.inf
for i in range(3):
    W_correct, W_wrong, max_error_cls = rankTest.find_prototype(data_point, lbl, kernel_size)
    rankTest.update_prot_and_omega(W_correct, W_wrong, lbl, max_error_cls, data_point)
    cost = rankTest._costfunc(data_point, lbl, kernel_size)
    print(cost)

# print("======")
# print("result: ", W_correct, W_wrong)
