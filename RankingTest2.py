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

    def find_prototype(self, list_dist, label, k_size):
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
    def update_prot_and_omega(self, w_plus, w_minus, label, max_error_cls, datapoint, protype_position):
        # print(w_plus, w_minus)
        while len(w_plus) > 0 and len(w_minus) > 0:
            # find closest correct prototype from w_plus
            min_value = np.inf
            min_ind = 0
            index = 0
            for prot in w_plus:
                if prot[1] < min_value:
                    min_value = prot[1]
                    min_ind = index
                index += 1
            closest_cor_p = w_plus.pop(min_ind)

            # find closest wrong prototype from w_minus
            min_value = np.inf
            min_ind = 0
            index = 0
            for prot in w_minus:
                if prot[1] < min_value:
                    min_value = prot[1]
                    min_ind = index
                index += 1
            closest_wro_p = w_minus.pop(min_ind)

            # update prototypes and omega here
            pt_pair = [closest_cor_p, closest_wro_p]
            self._derivatives(pt_pair, label, max_error_cls, datapoint, protype_position)
            # print(pt_pair)

    # calculate derivatives of prototypes a, b and omega
    def _derivatives(self, pt_pair, label, max_error_cls, datapoint, protype_position):
        # print(max_error_cls)
        # calculate alpha+ and alpha-
        distance_correct = pt_pair[0][1]
        distance_wrong = pt_pair[1][1]
        ranking_diff_correct = abs(label - pt_pair[0][0] // self.prototypes_per_class)
        ranking_diff_wrong = abs(label - pt_pair[1][0] // self.prototypes_per_class)

        alpha_plus = math.exp(- pow(ranking_diff_correct, 2) / (2 * pow(self.gaussian_sd, 2)))
        alpha_minus = math.exp(- pow(max_error_cls - ranking_diff_wrong, 2) / (2 * pow(self.gaussian_sd, 2))) *\
                math.exp(-pow(distance_wrong, 2) / (2 * pow((1-self.gaussian_sd), 2)))
        # print("alpha_plus, alpha_minus: ", alpha_plus, alpha_minus)

        alpha_distance_plus = alpha_plus * distance_correct
        alpha_distance_minus = alpha_minus * distance_wrong

        gamma_plus = 2*alpha_distance_minus / pow((alpha_distance_plus + alpha_distance_minus), 2)
        gamma_minus = -2*alpha_distance_plus / pow((alpha_distance_plus + alpha_distance_minus), 2)

        pid_correct = pt_pair[0][0]
        pid_wrong = pt_pair[1][0]
        diff_correct = datapoint - protype_position[pid_correct]
        diff_wrong = datapoint - protype_position[pid_wrong]

        diff_mtx_correct = diff_correct.T.dot(diff_correct)
        delta_omega_plus = gamma_plus * 2 * alpha_plus*self.omega_.dot(diff_mtx_correct)

        diff_mtx_wrong = diff_wrong.T.dot(diff_wrong)
        delta_omega_minus = gamma_minus * 2 * alpha_minus * self.omega_.dot(diff_mtx_wrong)

        delta_omega = delta_omega_plus + delta_omega_minus
        print("delta_omega:", delta_omega)

        delta_correct_prot = gamma_plus * (-2*alpha_plus*diff_correct.dot(self.omega_.T.dot(self.omega_)))
        delta_wrong_prot = gamma_minus * (-2*alpha_minus*diff_wrong.dot(self.omega_.T.dot(self.omega_)))
        print("delta:", delta_correct_prot, delta_wrong_prot)

    # def alpha_correct(self, rank_diff):
    #
    #
    #
    # def alpha_wrong(self, rank_diff, distance):


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
prot_position = np.array([[1.1, 3.5], [2.1, 2.8], [3.5, 1.1], [3.2, -1.7], [2.2, -2.3], [1.8, -3.4], [-1.4, -3.6]
                             , [-2.8, -2.2], [-3.1, -1.5], [-3.7, 1.9], [-2.9, 2.1], [-1.2, 3.4]])

l_distance =_squared_euclidean(prot_position, data_point).flatten()
# l_distance = np.array([4.11044186, 6.31040921, 58.6941359,  53.96639758, 32.68570672, 39.96803298,
#                        11.14983586,  8.34313266, 18.74119665, 11.66475519, 28.03848346, 29.49774315])
lbl = 1
W_correct, W_wrong, max_error_cls = rankTest.find_prototype(l_distance, lbl, 1)
rankTest.update_prot_and_omega(W_correct, W_wrong, lbl, max_error_cls, data_point, prot_position)
# print("======")
# print("result: ", W_correct, W_wrong)
