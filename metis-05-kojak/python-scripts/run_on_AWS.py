"""Runs the redistricting algorithm on AWS so I can close my computer."""
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
#subset = slice_up_a_state(data, long_east=-83)

# To prevent recurion depth limit on AWS.
sys.setrecursionlimit(10000)

# Stage One
k = [4/13, 4/13, 5/13]
pop_error_limit = 0.02
stage_one = redistricting(k=k, weights=[1, 1, 0], seed=42,
                          pop_error_limit=pop_error_limit,
                          compactness_method = 'sum',
                          gif=False, n_jobs=-1, logging=False,
                          verbose_time=True)
stage_one.fit(data)
save_pickle(stage_one, 'stage_one')
print('Succesfully pickled model.')

command = 'cp ../images/frames/* ../images/frames_stage_one'
subprocess.call(command.split())
print('Completed Stage One')

# Stage Two
dst_pops = [dst.current_pop for dst in stage_one.districts]
stage_two = []
for dst in stage_one.districts:
    if dst.current_pop == max(dst_pops):
        k = [2/5, 3/5]
        pop_error_limit = 0.005
    else:
        k=2
        pop_error_limit = 0.005

    model = redistricting(k=k, weights=[1, 1, 0], seed=42,
                          pop_error_limit=pop_error_limit,
                          compactness_method = 'sum',
                          gif=False, n_jobs=-1, logging=False,
                          verbose_time=True)
    model.fit(dst)
    stage_two.append(model)

save_pickle(stage_two, 'stage_two')
command = 'cp ../images/frames/* ../images/frames_stage_two'
subprocess.call(command.split())
print('Completed Stage Two')

# Stage Three
stage_two_dst = []
for model in stage_two:
    stage_two_dst += model.districts

dst_pops = [dst.current_pop for dst in stage_two_dst]
stage_three = []
for dst in stage_two_dst:
    if dst.current_pop == max(dst_pops):
        k = 3
        pop_error_limit = 0.005
    else:
        k = 2
        pop_error_limit = 0.005
    model = redistricting(k=k, weights=[1, 1, 0], seed=42,
                          pop_error_limit=pop_error_limit,
                          compactness_method = 'sum',
                          gif=False, n_jobs=-1, logging=False,
                          verbose_time=True)
    model.fit(dst)
    stage_three.append(model)

save_pickle(stage_three, 'stage_three')
command = 'cp ../images/frames/* ../images/frames_stage_three'
subprocess.call(command.split())
print('Completed Stage Three')

# Final Plot
_, ax = rpl.initialize_plot(stage_three_dst)
rpl.final_plot(ax, stage_three_dst, color_by_party=True)
