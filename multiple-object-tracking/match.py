import networkx as nx
import utils


class Matcher:
    def __init__(self):
        self.match_set = set()
        self.graph = nx.Graph()

    def match(self, state_list, det_list):
        """
        最大权值匹配
        :param state_list: 先验状态list
        :param det_list: 量测list
        :return: dict 匹配结果
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
