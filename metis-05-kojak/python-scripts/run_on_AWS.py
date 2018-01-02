"""
Runs the redistricting algorithm on AWS so I can close my computer.
"""
#Libraries
import json
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('/home/ubuntu/metis-05-kojak/python-scripts')
from redistricting import redistricting
from kojak import save_pickle
from kojak import slice_up_a_state

#Data
filepath = '../data/North-Carolina-2014.geojson'
with open(filepath) as f:
    data = json.load(f)
data = data['features']

#Subset the Data
#subset = slice_up_a_state(data, long_west=-81.2, long_east=-80)

# To prevent recurion depth limit on AWS.
sys.setrecursionlimit(10000)

#Model
model = redistricting(k=3, weights=[1, 1, 0], seed=42,
                      compactness_method = 'sum',
                      gif=False, n_jobs=-1, logging=True)
model.fit(data)
# model.initialize(data)
# model.grow_clusters(iter_limit=500)
# save_pickle(model, 'redistricting_model_500')
# model.grow_clusters(iter_limit=500)
# save_pickle(model, 'redistricting_model_1000')
# model.grow_clusters(iter_limit=500)
# save_pickle(model, 'redistricting_model_1500')
# model.grow_clusters(iter_limit=500)
# save_pickle(model, 'redistricting_model_2000')
# model.grow_clusters(iter_limit=500)
# model.quality_check()
# model.display_results()
save_pickle(model, 'state_3_')
save_pickle(model.districts, 'state_3_districts')
save_pickle(model.table, 'state_3_table')
print('Hey it actually finished!')
