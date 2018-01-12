"""
Runs the redistricting algorithm on AWS so I can close my computer.
"""
#Libraries
import json
import subprocess
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

# To prevent recurion depth limit on AWS.
sys.setrecursionlimit(10000)

# Stage One
k = [4/13, 4/13, 5/13]
stage_one = redistricting(k=k, weights=[1, 1, 0], seed=42,
                          compactness_method = 'sum',
                          gif=False, n_jobs=-1, logging=True)
stage_one.fit(data)

save_pickle(model, 'stage_one_')
command = 'cp ../images/frames/* ../images/frames_stage_one'
subprocess.call(command.split())
print('Completed Stage One')

# Stage Two
dst_pops = [dst.current_pop for dst in stage_one.districts]
stage_two_dst = []
for dst in stage_one.districts:
    if dst.current_pop == max(dst_pops):
        k = [2/5, 3/5]
    else:
        k=2

    model = redistricting(k=k, weights=[1, 1, 0], seed=42,
                          compactness_method = 'sum',
                          gif=False, n_jobs=-1, logging=True)
    model.fit(dst)
    stage_two_dst += model.districts

save_pickle(stage_two_dst, 'stage_two_dst_')
command = 'cp ../images/frames/* ../images/frames_stage_two'
subprocess.call(command.split())
print('Completed Stage Two')

# Stage Three
dst_pops = [dst.current_pop for dst in stage_two_dst]
stage_three_dst = []
for dst in stage_two_dst:
    if dst.current_pop == max(dst_pops):
        k = 3
    else:
        k = 2
    model = redistricting(k=k, weights=[1, 1, 0], seed=42,
                          compactness_method = 'sum',
                          gif=False, n_jobs=-1, logging=True)
    model.fit(dst)
    stage_three_dst += model.districts

save_pickle(stage_three_dst, 'stage_three_dst')
command = 'cp ../images/frames/* ../images/frames_stage_three'
subprocess.call(command.split())
print('Completed Stage Three')

# Final Plot
_, ax = rpl.initialize_plot(stage_two_dst)
rpl.final_plot(ax, stage_two_dst, color_by_party=True)
