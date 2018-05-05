import math
from math import log

import numpy as np


class RankingTest:
    ptype_id = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    prototypes_per_class = 2
    ranking_range = [0, 5]
    ranking_list = np.array(list(range(ranking_range[0], ranking_range[1] + 1)))

    def find_prototype(self, list_dist, label, k_size):
        print(list_dist, label, self.ptype_id)
        correct_idx0 = label * self.prototypes_per_class
        correct_idx1 = correct_idx0 + self.prototypes_per_class
        print(list_dist[correct_idx0:correct_idx1])

        self.ranking_list
        correct_cls_min = label - k_size
        correct_cls_max = label + k_size
        if correct_cls_min < self.ranking_range[0]:
            correct_cls_min = self.ranking_range[0]
        if correct_cls_max > self.ranking_range[1]:
            correct_cls_max = self.ranking_range[1]

        correct_ranking = np.array(list(range(correct_cls_min, correct_cls_max + 1)))
        print(correct_ranking)

        wrong_cls_min = self.ranking_range[0]
        wrong_cls_max = self.ranking_range[1]
        if wrong_cls_min == correct_cls_min:
            wrong_cls_min = correct_cls_max + 1
        if wrong_cls_max == correct_cls_max:
            wrong_cls_max = correct_cls_min - 1

        print(wrong_cls_min, wrong_cls_max)

        R_max = max(correct_cls_max - wrong_cls_min, wrong_cls_max - correct_cls_min)
        print(R_max)

        for i in range(R_max):
            r = i + 1
            print("pair R: ", r)
        print(self.ranking_list)


l_distance = np.array([4.11044186, 6.31040921, 58.6941359,  53.96639758, 32.68570672, 39.96803298,
                       11.14983586,  8.34313266, 18.74119665, 11.66475519, 28.03848346, 29.49774315])
lbl = 2
RankingTest().find_prototype(l_distance, lbl, 1)

