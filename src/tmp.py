# get from graph file
import json
with open('../data/gold_graph') as f:
    graphs = json.load(f)

for graph in graphs:
    nodes = graph['nodes']
    for node in nodes:
        if node['arg1'] != [-1] and node['arg2'] != [-1]:
            node['head'] = node['arg1'][0]
        elif node['arg1'] != [-1] and node['arg2'] != [-1]:
            nodes[node['arg1'][0]]['head'] = node['id']
            node['head'] = node['arg2'][0]

type_set = dict()
for graph in graphs:
    nodes = graph['nodes']
    for node in nodes:
        if type(node['arg1'])!=list and type(node['arg2'])!=list:
            new = (node['code'], nodes[node['arg1']]['code'] if node['arg1']!=-1 else -1, \
                          nodes[node['arg2']]['code'] if node['arg2']!=-1 else -1)
            if new not in type_set.keys():
                type_set[new] = 1
            else:
                type_set[new] = type_set[new] + 1

print(type_set)
