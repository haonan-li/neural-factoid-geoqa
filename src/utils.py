import re
import nltk
import json
import spacy
import sys, time
from tqdm import tqdm,tqdm_notebook
from allennlp.predictors.predictor import Predictor


PRONOUN = dict({'where':'1','what':'2','which':'3','when':'4','how':'5','whom':'6','whose':'6','why':'7','is':'8','are':'8','does':'8'})
POUNC = set(['.',',','\'','?','!','@','#','$','%','^','*','(',')','[',']',';','"',])
NUM_TAG = set(['CD'])
digit = set(['meter','meters','km','kilometer','kilometers','mile','miles'])
NOUN_TAG = set(['NN','NNS','NNP','NNPS'])
ADJ_TAG = set(['JJ','JJR','JJS'])
ADV_TAG = set(['RB','RBR','RBS'])
NO = set (['a','an','the','and','do','did','does','be','am','is','was','are','can','could','may','might','must','shall','should','will','would'])
VERB_TAG = set(['VB','VBD','VBG','VBN','VBP','VBZ'])
PREP_TAG = set(['IN','TO'])
ALL_NODE = set(['1','2','3','4','5','6','7','8','n','t','d','q','s','a','r','o','x'])


# load gazetteer in dictionary
def load_gazetteer(fga):
    fga = open(fga,'r')
    gazetteer = dict()
    for line in fga.readlines():
        line = line.strip()
        gazetteer[line.lower()] = line
    fga.close()
    return gazetteer

# load word
def load_word(fword):
    words = set()
    fword = open(fword,'r')
    for line in fword.readlines():
        word = line.strip()
        words.add(word)
    fword.close()
    return words

# load place type
def load_pt(fpt):
    pt_set = set()
    pt_dict = dict()
    fpt = open(fpt,'r')
    for line in fpt.readlines():
        word = line.strip().split()
        pt_set.add(word[1])
        pt_dict[word[0]] = word[1]
        pt_dict[word[1]] = word[1]
    fpt.close()
    return pt_set,pt_dict

# load spatial relation
def load_sp_relation(fspr):
    spr_dict = dict()
    fspr = open(fspr,'r')
    for line in fspr.readlines():
        line = line.strip().split('\t')
        spr_dict[line[0]] = line[1]
    fspr.close()
    return spr_dict

def load_data(fdata):
    fdata = open(fdata,'r')
    data = json.load(fdata)
    return data

def load_abbr(fabbr):
    abbr = dict()
    fabbr = open(fabbr,'r')
    for line in fabbr.readlines():
        line  = line.strip().split()
        abbr[line[-1]] = ' '.join(line[:-1])
    return abbr

# filter from dict 'ga' delete set 'topn'
def filterr(ga, topn):
    for word in topn:
        if word in ga:
            ga.pop(word)
    return ga

def process_data(data,ga,abbr,pt_set,pt_dict,actv,stav,sp_prep,spr_dict):
    for item in tqdm(data):
        ## process query
        query = item['query'].lower()
        analyze = process_sentence(query,ga,abbr,pt_set,pt_dict,actv,stav,sp_prep,spr_dict)
        item['queryAnalyze'] = analyze
        ## process answer
        if 'answers' in item:
            item['answersAnalyze'] = list()
            for answer in item['answers']:
                analyze = process_sentence(answer.lower(),ga,abbr,pt_set,pt_dict,actv,stav,sp_prep,spr_dict)
                item['answersAnalyze'].append(analyze)
    return data


class Node:
    def __init__(self, text, start, end, code):
        self.text = text
        self.start = start
        self.end = end
        self.code = code
        self.id = -1
        self.head = -2

class Instance:
    def __init__(self, question, ner_predictor=None, dep_predictor=None, const_predictor=None, gold_code=None, ga=None, abbr=None, pt_set=None, pt_dict=None, actv=None, stav=None, sp_prep=None, spr_dict=None):
        from nltk.stem.wordnet import WordNetLemmatizer
        self.sentence = ' '.join(question.strip().split()) # sentence may segmented by tab
        self.ner_predictor = ner_predictor
        self.dep_predictor = dep_predictor
        self.const_predictor = const_predictor

        self.dep_structure = self._get_dep()
        self.tokens = self.dep_structure['words']
        self.pos = self.dep_structure['pos']
        self.len = len(self.tokens)

        # Don't build graph when initialize
        if gold_code:
            self._from_gold(gold_code)
        else:
            self._encode(ga,abbr,pt_set,pt_dict,actv,stav,sp_prep,spr_dict)



    def _get_dep(self):
        return self.dep_predictor.predict(self.sentence)

    def __str__(self):
        return (f'sentence: {self.sentence} \
                \ncode: {self.code} \
                \nsimple code: {self.simple_code} \
                \nquestion word: {self.question_word} \
                \nplace name: {self.place_name} \
                \nplace type: {self.place_type} \
                \nspatial relation: {self.spatial_relation} \
                \ndigit: {self.digit} \
                \nquality: {self.quality} \
                \nobject: {self.object} \
                \nstative verb: {self.stative} \
                \nactivity verb: {self.activity}\n')

    def _encode(self, ga,abbr,pt_set,pt_dict,actv,stav,sp_prep,spr_dict):
        code = ['.' for i in range(self.len)]
        nodes = []
        self.detect_pn(code, nodes, ga, use_ner=True)
        self.detect_digit(code, nodes)
        self.detect_q_word(code, nodes)
        self.detect_pt(code, nodes, pt_dict)
        self.detect_relation(code, nodes, spr_dict)
        self.detect_obj_verb(code, nodes, actv, stav)
        self.detect_quality(code, nodes)

        self.code = code
        self.nodes = nodes

        self.node2statistic()
        self.simple_code = re.sub('[\.,-]','',(''.join(code)))
        self.build_node_dep_tree() # update nodes

    def _from_gold(self, code):
        token = self.tokens
        code = code.strip().split()
        nodes = []
        for j in range(len(code)):
            if code[j] in ALL_NODE:
                for k in range(j+1,len(code)):
                    if code[k] != '-':
                        break
                nodes.append(Node(' '.join(token[j:k]), j, k, code[j]))
        self.code = code
        self.nodes = nodes
        self.node2statistic()
        self.simple_code = re.sub('[\.,-]','',(''.join(code)))
        self.build_node_dep_tree() # update nodes

    def node2statistic(self):
        self.place_name = []
        self.digit = []
        self.question_word = []
        self.place_type = []
        self.spatial_relation = []
        self.object = []
        self.activity = []
        self.stative = []
        self.quality = []
        for node in self.nodes:
            if node.code == 'n':
                self.place_name.append((node.text, node.start, node.end))
            elif node.code== 't':
                self.place_type.append((node.text, node.start, node.end))
            elif node.code == 'd':
                self.digit.append((node.text, node.start, node.end))
            elif node.code == 'q':
                self.quality.append((node.text, node.start, node.end))
            elif node.code == 's':
                self.stative.append((node.text, node.start, node.end))
            elif node.code == 'a':
                self.activity.append((node.text, node.start, node.end))
            elif node.code == 'r':
                self.spatial_relation.append((node.text, node.start, node.end))
            elif node.code == 'o':
                self.object.append((node.text, node.start, node.end))
            else:
                self.question_word.append((node.text, node.start, node.end))

    def print_spr_tree(self):
        for node in self.nodes:
            print (vars(node))
        print ()

    def build_node_dep_tree(self):
        heads = self.build_dep_tree()
        # build map from token_id to node_id, -1 represents the node itself is root
        idx_to_node_idx = [-2] * len(heads)
        self.nodes = sorted(self.nodes,key=lambda x:x.start)
        for idx,node in enumerate(self.nodes):
            node.id = idx
            for i in range(node.start,node.end):
                idx_to_node_idx[i] = idx

        for idx,node in enumerate(self.nodes):
            for i in range(node.start,node.end):
                # heads[i] is the i's token's head token_id
                if heads[i] == -1:
                    node.head = -1
                elif idx_to_node_idx[heads[i]] != idx:
                    node.head = idx_to_node_idx[heads[i]]

    # Node deep tree for relation
    def build_dep_tree(self):
        pred_heads = [head-1 for head in self.dep_structure['predicted_heads']]
        my_heads = [-1] * len(pred_heads)
        # root token must be a node, make the dep 'nsubj' as root
        for index, head in enumerate(pred_heads):
            if head == -1 and self.code[index] in ['.',',']:
                for i,(h,dep) in enumerate(zip(pred_heads,self.dep_structure['predicted_dependencies'])):
                    if h==index and dep=='nsubj':
                        pred_heads[i] = -1
                        pred_heads[index] = i
                        break
        # my_heads will guarantee heads to other node or root word
        for index, head in enumerate(pred_heads):
            if head == -1:
                continue
            # head propagation (while head is not root and not a node)
            while pred_heads[head] != -1 and self.code[head] in ['.',',']:
                head = pred_heads[head]
            my_heads[index] = head
        return my_heads

    def detect_relation(self, code, nodes, spr_dict):
        # spatial relation (exact match)
        i = 0
        sp_relation = []
        for i,token in enumerate(self.tokens):
            if (code[i] in ['n','d','t'] and i > 0) or \
               (self.tokens[i]=='the' and i>0 and i+1<self.len and code[i+1] in ['n','t']):
                for j in range(0,i):
                    phrase = ' '.join(self.tokens[j:i])
                    if phrase in spr_dict.keys():
                        code[j] = 'r'
                        code[j+1:i] = ['-' for k in range(i-j-1)]
                        sp_relation.append((phrase,j,i))
                        nodes.append(Node(phrase,j,i,'r'))
                        i+=1
                        break

    def detect_digit(self, code, nodes):
        digit = []
        for i,token in enumerate(self.tokens):
            if self.pos[i] in NUM_TAG:
                if i+1<self.len and self.tokens[i+1] in digit:
                    code[i] = 'd'
                    code[i+1] = '-'
                    digit.append((' '.join(self.tokens[i:i+2]),i,i+2))
                    nodes.append(Node(' '.join(self.tokens[i:i+2]),i,i+2,'d'))
                elif re.match('^\d+k?m$',token) is not None:
                    code[i] = 'd'
                    digit.append((token,i,i+1))
                    nodes.append(Node(token,i,i+1,'d'))

    def detect_q_word(self, code, nodes):
        q_word = []
        for i,token in enumerate(self.tokens):
            # pronoun (is/are question must start with is/are)
            if (token.lower() in PRONOUN) and (i==0 or PRONOUN[token.lower()] != '8'):
                code[i] = PRONOUN[token.lower()]
                if token.lower() =='how' and i+1<self.len and self.pos[i+1] in (ADJ_TAG.union(ADV_TAG)):
                    code[i+1] = '-'
                    q_word.append((' '.join(self.tokens[i:i+2]),i,i+2))
                    nodes.append(Node(' '.join(self.tokens[i:i+2]),i,i+2,code[i]))
                else:
                    q_word.append((token,i,i+1))
                    nodes.append(Node(' '.join(self.tokens[i:i+1]),i,i+1,code[i]))

    def detect_pt(self, code,nodes,pt_dict):
        ptype = []
        for i,token in enumerate(self.tokens):
            if token in pt_dict:
                code[i] = 't'
                ptype.append((pt_dict[self.tokens[i]],i,i+1))
                nodes.append(Node(pt_dict[self.tokens[i]],i,i+1,'t'))

    def detect_pn(self, code, nodes, ga, use_ner):
        # noun phrases
        pname = []
        if use_ner:
            ner = self.ner_predictor.predict(self.sentence)
            st,ed = 0,0
            for i,tag in enumerate(ner['tags']):
                if tag == 'L-LOC':
                    ne = ' '.join(self.tokens[st:i+1])
                    code[st+1:i+1] = ['-' for k in range(i-st)]
                elif tag == 'U-LOC':
                    st = i
                    ne = self.tokens[i]
                    code[i] = 'n'
                elif tag == 'B-LOC':
                    st = i
                    code[i] = 'n'
                    continue
                else:
                    continue
                pname.append((ne, st, i+1))
                nodes.append(Node(ne,st,i+1,'n'))
        else:
            noun_phrases = extract_noun_phrases(self.sentence)
            i = 0
            while i < self.len:
                for j in range(self.len,i,-1):
                    phrase = ' '.join(self.tokens[i:j])
                    if phrase in noun_phrases and phrase in ga:
                        self.tokens[i:j] = ga[phrase].split()
                        code[i] = 'n'
                        code[i+1:j] = ['-' for k in range(j-i-1)]
                        pname.append((ga[phrase],i,j))
                        nodes.append(Node(ga[phrase],i,j,'n'))
                        i = j
                        break
                i += 1

    def detect_obj_verb(self, code, nodes, actv, stav):
        activity = []
        stative = []
        oobject = []

        for i,token in enumerate(self.tokens):
            # filter
            if token in NO:
                code[i] = ','
            elif (code[i] == '.'):
                # other object
                if (self.pos[i] in NOUN_TAG):
                    code[i] = 'o'
                    oobject.append((token,i,i+1))
                    nodes.append(Node(token,i,i+1,'o'))
                # verb (simple dict lookup)
                elif (self.pos[i] in VERB_TAG):
                    stem = WordNetLemmatizer().lemmatize(self.tokens[i],'v')
                    if stem in actv:
                        code[i] = 'a'
                        activity.append((token,i,i+1))
                        nodes.append(Node(token,i,i+1,'a'))
                    elif stem in stav:
                        code[i] = 's'
                        stative.append((token,i,i+1))
                        nodes.append(Node(token,i,i+1,'s'))

    def detect_quality(self, code, nodes):
        quality = []
        # quality
        for i,token in enumerate(self.tokens):
            # place quality
            if (code[i] == '.') and (self.pos[i] in ADJ_TAG) and (i+1<self.len) and (code[i+1] == 't'):
                code[i] = 'q'
                quality.append((token,i,i+1))
                nodes.append(Node(token,i,i+1,'q'))

# constitude parsing for NP
def extract_all_phrases(root):
    phrases = []
    phrases.append((root['nodeType'],root['word']))
    if 'children' in root:
        for child in root['children']:
            phrases += get_all_phrases(child)
    return phrases

def extract_noun_phrases(sentence):
    const_tree = const_predictor.predict(sentence=sentence)
    phrases = extract_all_phrases(const_tree['hierplane_tree']['root'])
    noun_phrases = set(map(lambda x: x[1], filter(lambda x: x[0] in ['NP','NN','NNP','NNS','NNPS'], phrases)))
    for phrase in list(noun_phrases):
        if phrase[:4]=='the ':
            noun_phrases.add(phrase[4:])
    return noun_phrases

#     delete punctuation mark
#     tokens = list(filter(lambda x: not x in POUNC, tokens))

#     # expand abbreviation
#     tokens = [abbr[token] if (token in abbr) else token for token in tokens]
#     tokens = ' '.join(tokens).split(' ')


def read_data_from_file(file_name, data_dir):
    # To use the predictors, please download them and replace the locations with your local location.
    ner_predictor = Predictor.from_path('/home/username/tools/allennlp_model/ner-model-2018.12.18.tar.gz')
    const_predictor = Predictor.from_path('/home/username/tools/allennlp_model/elmo-constituency-parser-2018.03.14.tar.gz')
    dep_predictor = Predictor.from_path("/home/username/tools/allennlp_model/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")

    fdata = f'{data_dir}/raw_data/dataset-v2.1-location-queries.json'
    fga = f'{data_dir}/gazetteer/gazetteer.txt'
    fabbr = f'{data_dir}/gazetteer/abbr.txt'
    fcommon_word = f'{data_dir}/common_word/top10000.txt'
    fpt = f'{data_dir}/place_type/place_type.txt'
    factv = f'{data_dir}/verb/action_verb.txt'
    fstav = f'{data_dir}/verb/stative_verb.txt'
    foutput = f'{data_dir}/result/hackfest_result.json'
    fsp_prep = f'{data_dir}/prep/prep.txt'
    fspr = f'{data_dir}/sp_relation/sp_relation.txt'

    print ('loading data ... ')
    pt_set,pt_dict = load_pt(fpt)
    actv = load_word(factv)
    stav = load_word(fstav)
    sp_prep = load_word(fsp_prep)
    abbr = load_abbr(fabbr)
    ga = load_gazetteer(fga)
    spr_dict = load_sp_relation(fspr)

    topn = load_word(fcommon_word)
    ga = filterr(ga,topn)
    print ('done')

    with open(file_name,'r') as fin:
        questions = fin.readlines()
        questions = list(map(lambda x: x.strip().split('\t')[-1], questions))
    results = []
    for question in tqdm(questions):
        instance = Instance(question,ner_predictor,dep_predictor,const_predictor,
                pt_set=pt_set, pt_dict=pt_dict, actv=actv, stav=stav, sp_prep=sp_prep,
                abbr=abbr, spr_dict=spr_dict, ga=ga)
        results.append(instance)
    return results


def load_gold_tag(file_name):
    dep_predictor = Predictor.from_path("/home/username/tools/allennlp_model/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")
    results = []
    with open(file_name,'r') as f:
        gold = f.readlines()
        for i in tqdm(range(int(len(gold)/2))):
            instance = Instance(gold[2*i], None, dep_predictor, None, gold[2*i+1])
            results.append(instance)
    return results


################## For Graph ###################

def load_gold_graph(file_name):
    with open(file_name) as f:
        return json.load(f)

def build_graph(nodes):
    node_level = dict()
    nodes_copy = nodes.copy()
    num_nodes = len(nodes)
    # top level nodes are nodes that need to find directly in the DB
    remove_id = []
    for node in nodes:
        if node['arg1'][0]==-1 and node['arg2'][0]==-1:
            node_level[node['id']] = 0
            remove_id.append(node['id'])
    nodes = list(filter(lambda x: x['id'] not in remove_id, nodes))

    while len(nodes)>0:
        remove_id = []
        for node in nodes:
            if node['arg1'][0] in node_level or node['arg2'][0] in node_level:
                if node['arg1'][0] in node_level and node['arg2'][0] in node_level:
                    node_level[node['id']] = max(node_level[node['arg1'][0]],node_level[node['arg2'][0]])+1
                    remove_id.append(node['id'])
                elif node['arg1'][0] in node_level and node['arg2'][0]==-1:
                    node_level[node['id']] = node_level[node['arg1'][0]]+1
                    remove_id.append(node['id'])
                elif node['arg2'][0] in node_level and node['arg1'][0]==-1:
                    node_level[node['id']] = node_level[node['arg2'][0]]+1
                    remove_id.append(node['id'])
        nodes = list(filter(lambda x: x['id'] not in remove_id, nodes))

    assert num_nodes == len(node_level)
    return node_level

def create_triple(nodes):
    node_level = build_graph(nodes)
    triples = []
    for i in range(5):
        for k,v in node_level.items():
            if v==i:
                if nodes[k]['arg1'][0] == -1:
                    node1 = nodes[k]
                    triples.append(f"{i} {node1['id']}:{node1['code']}:{node1['text']}")
                elif nodes[k]['arg1'][0] != -1 and nodes[k]['arg2'][0] == -1:
                    node1 = nodes[k]
                    node2 = nodes[nodes[k]['arg1'][0]]
                    triples.append(f"{i} {node1['id']}:{node1['code']}:{node1['text']} {node2['id']}:{node2['code']}:{node2['text']}")
                else:
                    node1 = nodes[nodes[k]['arg1'][0]]
                    node2 = nodes[k]
                    node3 = nodes[nodes[k]['arg2'][0]]
                    triples.append(f"{i} {node1['id']}:{node1['code']}:{node1['text']} {node2['id']}:{node2['code']}:{node2['text']} {node3['id']}:{node3['code']}:{node3['text']}")
    return triples



################## Data preprocess ###################

def split5fold(data_dir):
    import json
    # data_dir = '../data'
    with open(f'{data_dir}/gold_label.json') as f:
        data = json.load(f)

    label_set = []
    for piece in data:
        label_set += piece['code']
    label_set = set(label_set)
    label_map = {label:i for i,label in enumerate(label_set)}

    with open(f'{data_dir}/label2id.json','w') as f:
        json.dump(label_map,f)

    batch = int(len(data)/5)

    for i in range(5):
        with open(f'{data_dir}/cross/test{i}.json','w') as f:
            for piece in data[batch*i:batch*(i+1)]:
                piece = {'tokens':piece['sentence'].split(),'labels':list(map(lambda x: label_map[x],piece['code']))}
                f.write(json.dumps(piece)+'\n')
        with open(f'{data_dir}/cross/train{i}.json','w') as f:
            for piece in data[:batch*i]+data[batch*(i+1):]:
                piece = {'tokens':piece['sentence'].split(),'labels':list(map(lambda x: label_map[x],piece['code']))}
                f.write(json.dumps(piece)+'\n')




################## Data postprocess ###################
def node_statistic(results):
    root_t = []
    all_t = []
    edge_t = []
    for idx,result in enumerate(results):
        for node in result.nodes:
            if node.code.isdigit():
                node.code = 'num'
            all_t.append(node.code)
            if node.head==-1:
                root_t.append(node.code)
            else:
                edge_t.append((node.code, result.nodes[node.head].code))
                if node.code == 'r' and result.nodes[node.head].code == 'num':
                    result.print_spr_tree()
    print ('root:')
    for k in set(root_t):
        print (k,root_t.count(k))
    print ('all:')
    for k in set(all_t):
        print (k,all_t.count(k))
    print ('edge:')
    for k in set(edge_t):
        print (k,edge_t.count(k))
        print ()

def get_triplets(result):
    triplets = []
    dis_map = dict()
    simple_code = result['simple_code']
    nodes = result['nodes']
#     if simple_code[0]!='8' and simple_code[1]=='t':
#         triplets.append([nodes[0].code,nodes[1].text,None])
#     else:
#         triplets.append([nodes[0].code,None,None])
    for node in nodes[1:]:
        if node.code=='r':
            print (node.id)
            triplets.append([nodes[node.id-1].text, node.text, nodes[node.id+1].text])
    return triplets

def postprecess():
    # get labels from pred file
    import json
    pred_labels = json.load(open('../data/result/bert-tag.json')) # 100
    data = json.load(open('../data/gold_label.json')) # 200

    with open('../data/result/bert_label', 'w') as f:
        for item, pred_label in zip(data[100:],pred_labels):
            sentence = item['sentence']
            f.write(sentence+'\n')
            f.write('\t'.join(pred_label)+'\n')
    results = load_gold_tag('../data/result/bert_label')

    # convert gold_label to json_file and write
    data = []
    for result in results:
        data.append({'sentence':result.sentence,'code':result.code})
    with open('../data/result/bert_label.json','w') as f:
        json.dump(data,f,indent=4)

    # write to graph repsentation
    data = []
    for result in results:
        piece = dict()
        piece['sentence'] = result.sentence
        piece['nodes'] = []
        for node in result.nodes:
            piece['nodes'].append({'text': node.text, 'start': node.start, 'end': node.end, 'code': node.code, 'id': node.id, 'arg1':[-1], 'arg2':[-1]})
        data.append(piece)
    with open('../data/result/bert_graph','w') as f:
        json.dump(data,f, indent=4)

    a = set(list(map(lambda x: x.simple_code, results)))
    print(len(a))
    pattern = '^(((2|3)t(n|r|t|d)+)|((2|3)(n|r|t|d)+)|(1(n|t|r)+)|(8(n|t|r|d)+)|(5(n|r|t)+))$'
    matches = list(filter(lambda x: re.match(pattern,x.simple_code), results))
    not_matches = list(filter(lambda x: not re.match(pattern,x.simple_code), results))
    print (len(matches))

    ## json_result = []
    for match in matches:
        if 'nodes' in match:
            if match['nodes'][0].head == -3:
                continue
            else:
                triplets = get_triplets(match)
            del match['nodes']
            match['triplet'] = triplets
        for key in ['placeName','placeType','sp_relation','object']:
            match[key] = list(map(lambda x: x[0],match[key]))
        json_result.append(match)
    with open('triplet_result','w') as f:
        json.dump(json_result,f,indent=4)


def generate_triples():
    import json
    graphs = load_gold_graph('../data/gold_graph')
    data = load_gold_tag('../data/gold_label')

    for graph,instance in zip(graphs, data):
        nodes = graph['nodes']
        triples = create_triple(nodes)

        graph['triples'] = triples
        graph['code'] = instance.code
        graph['question word'] = instance.question_word
        graph['place name'] = instance.place_name
        graph['place type'] = instance.place_type
        graph['spatial relation'] = instance.spatial_relation
        graph['digit'] = instance.digit
        graph['quality'] = instance.quality
        graph['object'] = instance.object
        del graph['nodes']

    with open('../data/triples.json','w') as f:
        json.dump(graphs,f,indent=2)
