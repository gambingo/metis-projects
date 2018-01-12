"""
Uses the Constraint-Based Polygonal Spatial Clustring (CPSC) Algorithm
to split the provided geographic data into fair districts.

CPSC is defined in "Redistricting using Heuristic-Based Polygonal
districting," Deepti Joshi, Leen-Kiat Soh and Ashok Samal. 2009.

Terminology of attributes and methods taken from CPSC paper.

Designed with data of North Carolina. You may find a link to the source data in
the repo for this project: https://github.com/gambingo/metis-05-kojak
---
Metis Data Science Bootcamp
J. Gambino
November 2017
"""

import sys
import numpy as np
import pandas as pd
from math import pi
from warnings import warn
from datetime import datetime
from multiprocessing import Pool
from multiprocessing import cpu_count
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.geometry import Polygon
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon

from precinct import precinct
from precinct import calculate_precinct_score
from district import district
import redplotlib as rpl
from redplotlib import palette
from kojak import threshold
from kojak import wasted_votes


class redistricting:

    def __init__(self,
                 k=13,
                 weights=[1, 1, 0],
                 seed=0,
                 compactness_method = 'sum',
                 method_weights = [1, 1],
                 figsize=(10,10),
                 gif=False,
                 n_jobs=-1,
                 verbose_time=False,
                 verbose_score=False,
                 logging=False,
                 debug=False):
        """
        Uses the Constraint-Based Polygonal Spatial Clustring (CPSC) Algorithm
        to split the provided geographic data into fair districts.

        CPSC is defined in "Redistricting using Heuristic-Based Polygonal
        districting," Deepti Joshi, Leen-Kiat Soh and Ashok Samal. 2009.

        Terminology of attributes and methods taken from CPSC paper.
        ---
        k:          (int or list)
                    If int, the number of congressional districts to draw
                    If list, the proproportion of the state population to put
                    in each district. For example, k = [0.5, 0.25, 0.25]
        weights:    (list of ints or floats)
                    [w_population, w_compactness, w_competitiveness]
                    Weights of each heuristic function
        seed:       (int) State to seed the random generator
        compactness_method: (str) Method to use for calculating compactness.
                            Allowable methods are 'CSPC', 'circular',
                            'convexity', 'average', and 'sum'
        method_weights:     (list of ints or floats) [w_circular, w_convexity]
                            Weights of individual compactness metrics when
                            choosing compactness_method as 'circular' or
                            'convexity'
        figsize:    (tuple) figsize argument to pass to matplotlib.pyplot
        gif:        (bool) Whether or not to save the figure each iteration and
                    collect them into a gif when finished
        n_jobs      (int) Multiprocessing - number of CPUs. If -1, will split
                    work amongst all CPUs available
        verbose_time:       (bool) whether or not to print the time of each
                            iteration when running. Used for debugging.
        verbose_score:      (bool) whether or not to print the heuristic sub-
                            scores each iteration. Used for debugging.
        logging:    (bool) Whether to save print messages to a log file
        debug:      (bool) Turns on verbose printing, and prints precinct move
                    counts on the last frame.
        """
        if isinstance(k, int):
            self.k = k
            self.pop_proportions = None
        elif isinstance(k, list):
            margin = 0.0005
            if (sum(k) > 1+margin) or (sum(k) < 1-margin):
                raise Exception('Population proportions must sum to 1.')
            self.k = len(k)
            self.pop_proportions = iter(k)
        else:
            msg = 'k must be an integer or list of percentages that sum to 1.'
            raise TypeError(msg)

        self.weights = weights
        self.precincts = []
        self.districts = []

        if seed:
            np.random.seed(seed)

        self.move_limit = 2
        self.compactness_method = compactness_method
        self.method_weights = method_weights
        self.figsize = figsize
        self.gif = gif
        self.debug = debug

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.logging = logging
        if logging:
            self.verbose_time = True
            self.verbose_score = True
            print('Here we go!')
            self.old_output = sys.stdout
            filename = '../images/logs/log-file_{}.txt'.format(self.timestamp)
            self.log_file = open(filename, 'w')
            sys.stdout = self.log_file
            print('Heuristic Weights: ', weights)
        elif debug:
            self.verbose_time = True
            self.verbose_score = True
        else:
            self.verbose_time = verbose_time
            self.verbose_score = verbose_score

        if n_jobs == -1:
            self.n_jobs = cpu_count()
            # msg = ('Will split precinct scoring among {} CPUs.'
            #        '').format(self.n_jobs)
            # print(msg)
        else:
            self.n_jobs = n_jobs


    def verbose_print(self,  msg, flag):
        """
        Only print when the relevant flag is true
        ---
        Args:
            msg: (string) message to print
            flag (bool) either self.verbose_time or self.verbose_score
        """
        if flag:
            print(msg)


    def assign(self, pt, dstrct):
        """
        When a precinct is assigned to a district, the district object must
        receive a new member and the precinct object must receive a new
        assignment.

        Also, if that precinct has already been assigned, it must be removed
        from it's current district before it is added to a new district.
        """
        if pt.assigned:
            pt.dstrct.remove_member(pt)
            pt.move_count += 1

        if pt.move_count>2:
            msg = ("Warning: You're adding a polygon with a move"
                   " count of {}.").format(pt.move_count)
            warn(msg)

        pt.assign_to_district(dstrct)     # assign a district to a precinct
        dstrct.add_member(pt)            # assign a precinct to a district


    def calculate_population_stats(self, data):
        """
        Loop through each precinct to calculate various population statistics
        needed to run the algorithm.
        ---
        Args:
            data:   dictionary of voting tabulation districts (self.precincts)
                    data = {geometry: dictionary of coordinates,
                            properties: dictionary of population stats}
        """
        self.state_pop = 0
        greatest_population = 0
        least_population = 323.1 * 10**6 #pop of the USA

        # Raw GeoJSON or previously drawn macro-district
        if isinstance(data, list):
            for pt in data:
                pop = pt['properties']['population']
                self.state_pop += pop

                if pop > greatest_population:
                    greatest_population = pop
                if pop < least_population:
                    least_population = pop

        elif isinstance(data, district):
            self.state_pop = data.current_pop
            for pt in data.members:
                pop = pt.properties['population']
                if pop > greatest_population:
                    greatest_population = pop
                if pop < least_population:
                    least_population = pop
        else:
            msg = 'Data must be a list of GeoJSON objects or a district object.'
            raise TypeError(msg)

        self.x = self.state_pop/self.k
        return (self.x, greatest_population, least_population)


    def are_neighbors(self, pt1, pt2):
        """
        Return whether or not two precincts are neighbors.
        1. Do they touch
        2. Does their combination result in a Polygon (not a MultiPoylgon)
        """
        if not pt1.geometry.touches(pt2.geometry):
            return False
        else:
            test_district = unary_union([pt1.geometry, pt2.geometry])
            return isinstance(test_district, Polygon)


    def find_precinct_neighbors(self, pt):
        """
        Find all neigboring self.precincts for a given district.
        """
        nbrs = [p for p in self.precincts if self.are_neighbors(pt, p)]
        pt.neighbors = nbrs


    def define_neighboorhoods(self):
        """
        Each precinct has known neighboring precincts. Find and store them at
        the beginnging of the algorithm. This helps define a "neighborhood"
        that will make finding district neighbors much faster later on.
        """
        print('Defining neighborhoods...')
        t = datetime.now()
        for pt in self.precincts:
            self.find_precinct_neighbors(pt)
        msg = 'Took {} to define the neighborhoods.'.format(datetime.now()-t)
        #self.verbose_print(msg, self.verbose_time)
        print(msg)


    def intitialzize_precincts(self, data):
        """
        For each precinct in the data, make one or more precinct objects.
        Certain data points contain multiple polgons. These must be split
        into multiple sub-self.precincts.
        ---
        Args:
            data:   dictionary of voting tabulation districts (self.precincts)
                    data = {geometry: dictionary of coordinates,
                            properties: dictionary of population stats}
        """
        pop_stats = self.calculate_population_stats(data)

        # Raw GeoJSON or previously drawn macro-district
        if isinstance(data, list):
            # Check geometry type
            for pt in data:
                shp = shape(pt['geometry'])
                if isinstance(shp, Polygon):
                    single_pt = precinct(pt, pop_stats, self.weights,
                                         self.compactness_method,
                                         self.method_weights)
                    self.precincts.append(single_pt)
                elif not isinstance(shp, Polygon):
                    if len(shp) == 1:
                        # Many datapoints are a MultiPolygon made of one Polygon
                        sub_pt = precinct(pt, pop_stats, self.weights,
                                          self.compactness_method,
                                          self.method_weights)
                        sub_pt.geometry = sub_pt.geometry[0]
                        self.precincts.append(sub_pt)
                    else:
                        # Must split each precinct into sub-self.precincts
                        A = [poly.area for poly in shp]
                        A = [a/sum(A) for a in A]

                        for ii, poly in enumerate(shp):
                            # Divvy up population/voting totals by area
                            sub_pt = precinct(pt, pop_stats, self.weights,
                                              self.compactness_method,
                                              self.method_weights)
                            sub_pt.geometry = poly
                            for prop in sub_pt.properties:
                                if isinstance(sub_pt.properties[prop], int):
                                    partial = sub_pt.properties[prop]/A[ii]
                                    sub_pt.properties[prop] = partial
                            self.precincts.append(sub_pt)
                else:
                    msg = 'Geometry data is not of type Polygon or MultiPolygon'
                    raise Exception(msg)
            msg = ('{} data points turned into {} self.precincts.'
                   '').format(len(data), len(self.precincts))
            print(msg)

            # Repair invalid geometry
            invalid_polygons = 0
            zero_area_polygons = 0
            for pt in self.precincts:
                if not pt.geometry.is_valid:
                    pt.geometry = pt.geometry.buffer(0)
                    invalid_polygons += 1

            original_length = len(self.precincts)
            self.precincts = [p for p in self.precincts if p.geometry.area>10**-7]
            new_length = len(self.precincts)

            if invalid_polygons > 0:
                msg = ('Repaired {} self.precincts with '
                       'invalid geometry.').format(invalid_polygons)
                print(msg)
                # msg = ('There are {} self.precincts with invalid geometry. '
                #        'They have not been repaired.').format(invalid_polygons)
                # warn(msg)
            msg = ('{} polygons were tossed for having zero area. That leaves {} '
                   'self.precincts.').format(original_length-new_length, new_length)
            print(msg)

            # In the NC dataset, some of the self.precincts were MultiPolygons of
            # length two, with the second polygon having zero area and length
            for pt in self.precincts:
                if isinstance(pt.geometry, MultiPolygon):
                    pt.geometry = pt.geometry[0]

        elif isinstance(data, district):
            self.precincts = data.members
            for pt in self.precincts:
                pt.reset(pop_stats)
        else:
            msg = 'Data must be a list of GeoJSON objects or a district object.'
            raise TypeError(msg)

        self.define_neighboorhoods()
        self.fig, self.ax = rpl.initialize_plot(self.precincts,
            figsize = self.figsize,
            weights = self.weights,
            compactness_method = self.compactness_method)
        self.num_precincts = len(self.precincts)


    def dist_to_seeds(self, pt):
        """
        Euclidean distance between precinct and closest seed precinct.
        Args:
            pt: precinct object
        """
        distances = []
        a = np.array(pt.centroid)
        for dstrct in self.districts:
            b = np.array(dstrct.centroid)
            distances.append(np.linalg.norm(a-b))
        return min(distances)


    def select_seeds(self, data):
        """
        1. Initialize precinct classes
        2. Determine target population for districts
        3. Choose seed self.precincts by K++ Initialization
        4. Assign to self.districts
        """
        self.intitialzize_precincts(data)

        seed_pt = np.random.choice(self.precincts)
        color_num = 0
        new_dstrct = district(self.weights,
                            self.compactness_method,
                            self.method_weights,
                            palette[color_num])
        if self.pop_proportions:
            new_dstrct.x = self.state_pop * next(self.pop_proportions)
        color_num += 1
        self.assign(seed_pt, new_dstrct)
        self.districts.append(new_dstrct)

        while len(self.districts) < self.k:
            # Compute distances
            distances = [self.dist_to_seeds(pt) for pt in self.precincts]
            probs = [x**2 for x in distances]
            sum_x = sum(probs) # random.choice requires probs to sum to one
            probs = [float(x)/sum_x for x in probs]
            # randomly select additional seed
            seed_pt = np.random.choice(self.precincts, p=probs)
            new_dstrct = district(self.weights,
                                self.compactness_method,
                                self.method_weights,
                                palette[color_num])
            if self.pop_proportions:
                new_dstrct.x = self.state_pop * next(self.pop_proportions)
            color_num += 1
            self.assign(seed_pt, new_dstrct)
            self.districts.append(new_dstrct)

        # rpl.plot_seeds(self.ax, self.districts)
        rpl.final_plot(self.ax, self.districts, iteration=self.iteration)
        print('Selected seeds.')


    def initialize(self, data):
        """
        Take in data.
        Initialize self.precincts and self.districts
        ---
        Args:
            data:   dictionary of voting tabulation districts (self.precincts)
                    data = {geometry: dictionary of coordinates,
                            properties: dictionary of population stats}
        """
        self.start = datetime.now()
        self.iteration = 0
        self.select_seeds(data)


    def find_district_neighbors(self, dstrct):
        """Find all neighboring precincts for a given district."""
        #t = datetime.now()

        potential_nbrs = []
        for pt in dstrct.members:
            potential_nbrs += pt.neighbors

        potential_nbrs = set(potential_nbrs)
        nbrs = [pt for pt in potential_nbrs if pt not in dstrct.members]

        if len(nbrs) == 0:
            msg = ('We got a problem. A district has absolutely no neighbors. '
                   'This can happen if a district has consumed the whole state '
                   'or if neighbor assignment failed.')
            raise Exception(msg)

        dstrct.neighbors = nbrs


    def select_best_district(self):
        """
        Select the district that needs the most help.
        This will be the district with the largest value of F().
        """
        needs_nbrs = [cl for cl in self.districts if cl.regenerate_neighbors]
        for dstrct in needs_nbrs:
            self.find_district_neighbors(dstrct)

        self.districts = sorted(self.districts, key=lambda x: x.F(),
                                reverse=True)
        self.verbose_print('Eligible Districts - Heuristic Score',
                           self.verbose_score)
        self.verbose_print([x.F() for x in self.districts], self.verbose_score)
        self.verbose_print('Eligible Districts - Population Heuristic:',
                           self.verbose_score)
        self.verbose_print([x.H() for x in self.districts], self.verbose_score)
        self.verbose_print('Eligible Districts - Compactness Heuristic:',
                           self.verbose_score)
        self.verbose_print([x.G() for x in self.districts], self.verbose_score)
        self.verbose_print('Eligible Districts - Gap Heuristic:',
                           self.verbose_score)
        self.verbose_print([x.E() for x in self.districts], self.verbose_score)
        self.verbose_print('\n', self.verbose_score)

        # Last set new neighbor flag to False for all clus   ters.
        # (precinct assignment or removal sets this flag to True)
        for dstrct in self.districts:
            dstrct.regenerate_neighbors = False

        return self.districts[0]


    def valid_removal(self, pt):
        """
        If the removal of a precinct from a district will split that district
        into two non-contiguous districts, that precinct cannot be removed.
        """
        if pt.assigned:
            current_dstrct = pt.dstrct
            if len(current_dstrct.members) <= 1:
                # Removal would yield an empty district
                return False
            else:
                test_district = current_dstrct.geometry.difference(pt.geometry)
                # if isinstance(test_district, MultiPolygon):
                if not isinstance(test_district, Polygon):
                    # Removal would split a district
                    return False
        return True


    def score_precincts(self, eligible_nbrs, best_dstrct):
        """Calculates each precinct's score relative to the best district"""
        # MultiProcessing
        p = Pool(processes=self.n_jobs)
        args = zip(eligible_nbrs, [best_dstrct]*len(eligible_nbrs))
        scores = p.starmap(calculate_precinct_score, args)
        nbrs_and_scores = zip(eligible_nbrs, scores)
        p.terminate()

        nbrs_and_scores = sorted(nbrs_and_scores,
                                 key=lambda x: x[1],
                                 reverse=True)
        self.verbose_print('Eligible Neighbors - Heuristic Score',
                           self.verbose_score)
        self.verbose_print([x.F(best_dstrct) for x in eligible_nbrs[0:5]],
                           self.verbose_score)
        self.verbose_print('Eligible Neighbors - Population Heuristic:',
                           self.verbose_score)
        self.verbose_print([x.H(best_dstrct) for x in eligible_nbrs[0:5]],
                           self.verbose_score)
        self.verbose_print('Eligible Neighbors - Compactness Heuristic:',
                           self.verbose_score)
        self.verbose_print([x.G(best_dstrct) for x in eligible_nbrs[0:5]],
                           self.verbose_score)
        self.verbose_print('Eligible Neighbors - Gap Heuristic:',
                           self.verbose_score)
        self.verbose_print([x.E(best_dstrct) for x in eligible_nbrs[0:5]],
                           self.verbose_score)
        self.verbose_print('\n', self.verbose_score)

        best_pt = None
        for pt, _ in nbrs_and_scores:
            if self.valid_removal(pt):
                best_pt = pt
                break
            else:
                print("This precinct is a cornerstone precinct. Try again.")

        return best_pt


    def select_best_precinct(self, best_dstrct):
        """
        For the chosen district, choose the precinct that can best help it.
        This will be the precinct with the highest value of F().
        This is also where we could include any intra/inter-district constraints
        but none are implemented at this time.
        """
        # First, look at all neighbors with a valid move count
        eligible = lambda pt: pt.move_count<self.move_limit
        eligible_nbrs = [pt for pt in best_dstrct.neighbors if eligible(pt)]

        if len(eligible_nbrs) == 0:
            # If no neighbors have a valid move count, take them all
            eligible_nbrs = best_dstrct.neighbors
            # Reset move counts in order to prevent oscillations
            for pt in eligible_nbrs:
                pt.move_count = self.move_limit-1

        best_pt = self.score_precincts(eligible_nbrs, best_dstrct)
        if best_pt == None:
            # All precinct's with a valid move count were cornerstone precincts.
            # Recalculate precinct scores for all neighbors.
            # This repeats computation, but is so rare that it is faster than
            # calculating for valid removal earlier.
            # The more compact districts are, the less this will happen.
            nbrs = [p for p in best_dstrct.neighbors if self.valid_removal(p)]
            eligible_nbrs = [pt for pt in nbrs if eligible(pt)]
            if len(eligible_nbrs) == 0:
                eligible_nbrs = nbrs
                for pt in eligible_nbrs:
                    pt.move_count = self.move_limit-1

            best_pt = self.score_precincts(eligible_nbrs, best_dstrct)

        if best_pt == None:
            # If this is still None, we have a big problem. This is unexpected.
            # If it were to happen, this district should be passed for now.
            rpl.final_plot(self.ax, self.districts, iteration=self.iteration)
            raise Exception('All neighbors are cornerstone precincts')

        return best_pt


    def grow_one_district(self):
        """
        1. Select best district to grow
        2. Select best precinct to add to the district
        3. Add the precinct to the district
        """
        t = datetime.now()
        best_dstrct = self.select_best_district()
        msg = "Took {} to find the best district.".format(datetime.now()-t)
        self.verbose_print(msg, self.verbose_time)

        #rpl.highlight_district(self.ax, best_dstrct, self.iteration)

        t = datetime.now()
        best_pt = self.select_best_precinct(best_dstrct)
        msg = "Took {} to find the best precinct.".format(datetime.now()-t)
        self.verbose_print(msg, self.verbose_time)

        self.assign(best_pt, best_dstrct)


    def state_efficiency_gap(self):
        """
        Calculates net wasted votes across all districts to yield the state-
        wide efficiency gap.
        """
        rep_total_wasted = 0
        dem_total_wasted = 0
        total_votes = 0
        for dstrct in self.districts:
            thresh = threshold(dstrct.rep_votes + dstrct.dem_votes)
            rep_wasted = wasted_votes(dstrct.rep_votes, thresh)
            dem_wasted = wasted_votes(dstrct.dem_votes, thresh)
            rep_total_wasted += rep_wasted
            dem_total_wasted += dem_wasted
            total_votes += dstrct.rep_votes + dstrct.dem_votes

        gap = (rep_total_wasted - dem_total_wasted)/total_votes
        return abs(gap)


    def passed_quality_thresholds(self):
        """Check quality thresholds for each criterion"""
        if self.iteration < self.num_precincts:
            return False
        else:
            assigned_pts = all([pt.assigned for pt in self.precincts])

        if self.weights[0]:
            # pop = all([e<0.01 for e in self.pop_percent_error()])
            pop = all([abs(dstrct.H_)<0.01 for dstrct in self.districts])
        else:
            pop = True

        if self.weights[2]:
            # A weight for the efficiency gap has been defined
            # gap = all([dstrct.gap<0.08 for dstrct in self.districts])
            gap = self.state_efficiency_gap() < 0.08
        else:
            gap = True

        return all([assigned_pts, pop])


    def grow_districts(self, iter_limit=None):
        """
        Grow self.districts from their current state. This can happen
        immediately after calling self.initalize() or after the model has been
        paused.
        ---
        KWargs:
            iter_limit: (int) how many iterations to run. If None or 0, will
                        run until stopping condition.
                        Note:   self.iteration counts how many total cycles the
                                model has run. local_iter counts how many cycles
                                the model has run in one instance of calling
                                self.grow_districts()
        """
        # Status Message
        if iter_limit:
            msg = ('Growing {} districts for {} iterations.'
                   '').format(self.k, iter_limit)
        else:
            msg = ('Growing {} districts till stopping condition...'
                   '').format(self.k)
        print(msg)

        # Grow self.districts
        local_iter = 0
        while not self.passed_quality_thresholds():
            print('Iteration: {}'.format(local_iter))
            self.grow_one_district()

            if self.gif:
                rpl.update_plot(self.ax, self.districts[0], self.iteration)

            self.iteration += 1
            local_iter += 1

            if iter_limit:
                if local_iter >= iter_limit:
                    break

            if self.iteration%50 == 0:
                rpl.final_plot(self.ax, self.districts, iteration=self.iteration)


    def results_table(self):
        """Create and display a pandas dataframe summarizing the districts"""
        df = pd.DataFrame(columns = ['Population',
                                     'Pop. Percent Error',
                                     'Compactness Score',
                                     'Efficiency Gap',
                                     'Heuristic Func'])
        df.index.name = 'District'
        df['Population'] = [dstrct.current_pop for dstrct in self.districts]
        errors = [abs(dstrct.H_) for dstrct in self.districts]
        df['Pop. Percent Error'] = [str(round(e*100, 2))+'%' for e in errors]
        if self.weights[1]:
            df['Compactness Score'] = [dstrct.G_ for dstrct in self.districts]
        if self.weights[2]:
            df['Efficiency Gap'] = [dstrct.gap for dstrct in self.districts]
        df['Heuristic Func'] = [dstrct.F_ for dstrct in self.districts]
        return df


    def display_results(self):
        """
        1. Combine plots into a .gif
        2. Print execution time
        3. Display results table
        """
        self.elapsed_time = datetime.now() - self.start
        msg = 'Completed in {}.'.format(self.elapsed_time)
        print(msg)

        if self.gif:
            t = datetime.now()
            print('Making gif...')
            rpl.make_gif(self.timestamp, self.iteration)
            print('Took {} to make the gif.'.format(datetime.now()-t))
        else:
            rpl.final_plot(self.ax, self.districts, iteration=self.iteration)
            rpl.final_plot(self.ax, self.districts, iteration=self.iteration,
                           color_by_party=True)

        self.table = self.results_table()
        print('El Fin')

        if self.logging:
            sys.stdout = self.old_output
            self.log_file.close()
            print('It finished!')

        return self.table


    def fit(self, data, iter_limit=None):
        """
        1. Initialize
        2. Grow districts
        3. Make gif
        """
        self.initialize(data)
        self.grow_districts(iter_limit=iter_limit)
        self.display_results()


    def quality_check(self):
        """
        Various checks to make sure the algorithm did an ok job
        """
        mp, p, other = 0, 0, 0
        for pt in self.precincts:
            if isinstance(pt.geometry, MultiPolygon):
                mp += 1
            elif isinstance(pt.geometry, Polygon):
                p += 1
            else:
                other += 1
        msg = ('There are {} multipolygon precincts, {} polygon '
               'precincts, and {} other types.').format(mp, p, other)
        print(msg)

        mp, p, other = 0, 0, 0
        for dstrct in self.districts:
            if isinstance(dstrct.geometry, MultiPolygon):
                mp += 1
            elif isinstance(dstrct.geometry, Polygon):
                p += 1
            else:
                other += 1

        msg = ('There are {} multipolygon districts, {} polygon '
               'districts, and {} other types.').format(mp, p, other)
        print(msg)
        msg = 'There are {} eligible districts.'.format(len(self.districts))
        print(msg)
