import networkx as nx
import utils


class Matcher:
    def __init__(self):
        self.match_set = set()
        self.graph = nx.Graph()

    def match(self, state_list, det_list):
        """
        Maximum weight matching
        :param state_list: A priori state list
        :param det_list: Measurement list
        :return: dict Matching results
        """
        for idx_state, state in enumerate(state_list):
            state_node = 'state_%d' % idx_state
            self.graph.add_node(state_node, bipartite=0)
            for idx_mea, det in enumerate(det_list):
                det_node = 'det_%d' % idx_mea
                self.graph.add_node(det_node, bipartite=1)
                score = utils.cal_iou(state, det)
                if score is not None:
                    self.graph.add_edge(state_node, det_node, weight=score)
        self.match_set = nx.max_weight_matching(self.graph)
        res = dict()
        for (node_1, node_2) in self.match_set:
            if node_1.split('_')[0] == 'det':
                res[node_2] = node_1
            else:
                res[node_1] = node_2
        return res
