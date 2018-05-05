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
        print("correct_ranking: ", correct_ranking)
        print(self.ranking_list)

        wrong_cls_min = self.ranking_range[0]
        wrong_cls_max = self.ranking_range[1]
        if wrong_cls_min == correct_cls_min:
            wrong_cls_min = correct_cls_max + 1
        if wrong_cls_max == correct_cls_max:
            wrong_cls_max = correct_cls_min - 1

        print(wrong_cls_min, wrong_cls_max)
        class_list = np.zeros((len(self.ptype_id)//self.prototypes_per_class), dtype=bool)
        class_list[correct_ranking] = True

        R_max = max(correct_cls_max - wrong_cls_min, wrong_cls_max - correct_cls_min)
        print(R_max)

        all_ptype_pairs = np.array([], dtype=np.int32)
        all_rank_pairs = np.array([], dtype=np.int32)
        for i in range(R_max):
            r = i + 1
            print("pair R: ", r)
            for c in correct_ranking:
                if c - r >= self.ranking_range[0] and not class_list[c - r]:
                    toAdd = np.array([c - r, c], dtype=np.int32)
                    all_rank_pairs = np.append(all_rank_pairs, toAdd)
                if c + r <= self.ranking_range[1] and not class_list[c + r]:
                    toAdd = np.array([c, c + r], dtype=np.int32)
                    all_rank_pairs = np.append(all_rank_pairs, toAdd)
        all_rank_pairs = all_rank_pairs.reshape(len(all_rank_pairs) // 2, 2)
        print("all_rank_pairs: ", all_rank_pairs)

        for p in all_rank_pairs:
            idx0 = p[0]*self.prototypes_per_class
            idx1 = p[0] * self.prototypes_per_class + self.prototypes_per_class
            min0 = np.inf
            min_id0 = -1
            for prot_idx in range(idx0, idx1):
                p_dist = list_dist[prot_idx]
                if p_dist < min0:
                    min0 = p_dist
                    min_id0 = prot_idx

            idx0 = p[1] * self.prototypes_per_class
            idx1 = p[1] * self.prototypes_per_class + self.prototypes_per_class
            min1 = np.inf
            min_id1 = -1
            for prot_idx in range(idx0, idx1):
                p_dist = list_dist[prot_idx]
                if p_dist < min1:
                    min1 = p_dist
                    min_id1 = prot_idx

            toAdd = np.array([min_id0, min_id1], dtype=np.int32)
            all_ptype_pairs = np.append(all_ptype_pairs, toAdd)
        all_ptype_pairs = all_ptype_pairs.reshape(len(all_ptype_pairs) // 2, 2)
        print("all_ptype_pairs", all_ptype_pairs)
        return all_ptype_pairs


l_distance = np.array([4.11044186, 6.31040921, 58.6941359,  53.96639758, 32.68570672, 39.96803298,
                       11.14983586,  8.34313266, 18.74119665, 11.66475519, 28.03848346, 29.49774315])
lbl = 2
res = RankingTest().find_prototype(l_distance, lbl, 1)
# lbl = 2
# RankingTest().find_prototype(l_distance, lbl, 1)
# lbl = 4
# RankingTest().find_prototype(l_distance, lbl, 1)
print("======")
print("result: ", res)
