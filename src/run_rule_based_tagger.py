# This file is used to transfer the result
from utils import *
agile_results = read_data_from_file('../data/questions201', '../place-qa-AGILE19/data')

# output agile_results
with open('../data/agile-tag.json','w') as f:
    out = []
    for r in result:
        out.append(r.code)
    json.dump(out,f,indent=4)
