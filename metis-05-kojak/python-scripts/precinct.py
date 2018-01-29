import numpy as np
from math import pi
from shapely.geometry import shape
from shapely.ops import unary_union

# import sys
# path = '/Users/Joe/Documents/Metis/Projects/metis-05-kojak/python-scripts/'
# sys.path.append(path)
from kojak import threshold
from kojak import wasted_votes


def calculate_precinct_score(pt, dstrct):
    """
    Updates the pt.F_ attribute in regards to the given district.
    ---
    Previously, this was a mere lambda. But it needs to be a top level
    function to be passed to a multiprocessing object.
    """
    return pt.F(dstrct)


class precinct:

    def __init__(self, data, pop_stats, weights,
                 compactness_method,
                 method_weights):
        """
        Stores information for an individual polygon (precinct).
        ---
        Args
            data: GeoJSON data of a single precinct
        """
        self.properties = data['properties']
        self.population = int(round(data['properties']['population']))
        # When multipolygon precincts are split into multiple precincts, all
        # stats are proportioned by area. This leads to non-integer populations 
        self.geometry = shape(data['geometry'])
        self.x = pop_stats[0]                   #target pop for districts
        self.greatest_pop = pop_stats[1]
        self.least_pop = pop_stats[2]

        # Sub-self.precincts will share centriods, but that's ok.
        self.centroid = self.geometry.centroid.bounds[0:2]
        self.assigned = False
        self.move_count = 0
        self.dstrct = None
        #self.F_ = None

        # Heuristic Functions
        # self.H_, self.G_, self.E_, self.F_ = None, None, None, None

        # Weights for each heuristic function
        self.w_H = weights[0] #population
        self.w_G = weights[1] #compactness
        self.w_E = weights[2] #efficiency gap

        self.compactness_method=compactness_method
        self.w_R = method_weights[0]
        self.w_C = method_weights[1]

        self.neighbors = []


    def reset(self, pop_stats):
        """
        Reset all attributes. Used when drawing districts from a larger macro-
        district. Initiale a blank map.
        """
        self.assigned = False
        self.move_count = 0
        self.dstrct = None

        self.x = pop_stats[0]                   #target pop for districts
        self.greatest_pop = pop_stats[1]
        self.least_pop = pop_stats[2]


    def assign_to_district(self, dstrct):
        """Each precinct must know what district it belongs to"""
        self.dstrct = dstrct


    def H(self, dstrct):
        """
        Heuristic function that calculates precinct's influence on a
        district's population
        ---
        This is normalized so that the most populous precinct has a score
        of one and the least populous has score of zero.
        """
        new_pop = self.population + dstrct.current_pop
        score = (self.x - new_pop)/self.x
        # Here the absolute value tries to prevent a district that only
        # needs a minor increase in population taking the most populous pt.
        score = 1-abs(score)
        self.H_ = self.w_H * score
        return self.H_


    def CSPC_score(self, dstrct):
        """
        Compactness Method 1:
        The CSPC specifies the following method. The seems to work well for
        regular shaped self.precincts like those of Nebraska, but it doesn't
        generalize well to the irregular shaped self.precincts of North
        Carolina.

        For each unassigned neighboring polygon, calculate:
            (overlap + new_border)/overlap
        where:
            overlap =   length of district border shared with unassigned
                        neighbors
            new_border= length of the district border if this polygon were
                        added
        """
        overlap = self.geometry.intersection(dstrct.geometry).length
        # smoothing; if they share a corner, the overlap is zero
        overlap += 0.000001
        candidate_dstrct = unary_union([self.geometry, dstrct.geometry])
        new_border = candidate_dstrct.length
        return (overlap+new_border)/overlap


    def circular_score(self, dstrct):
        """
        Compactness Method 2:
        A second method is to calculate the ratio of border length to
        polygon area. The ratio is normalized to return 1 for a circle.
        """
        candidate_dstrct = unary_union([self.geometry, dstrct.geometry])
        ratio = 4*pi*candidate_dstrct.area/(candidate_dstrct.length**2)
        return ratio


    def convexity_score(self, dstrct):
        """
        Compactness Method 3:
        The convexity ratio measures the ratio of a shape's area to the area
        of the smallest convex shape that contains all points in the
        shape. It equals one for a fully convex shape.
        """
        candidate_dstrct = unary_union([self.geometry, dstrct.geometry])
        A_shape = candidate_dstrct.area
        A_hull  = candidate_dstrct.convex_hull.area
        convexity = A_shape/A_hull
        return convexity


    def G(self, dstrct):
        """
        Heuristic function controlling compactness of a district. There are
        several methods for scoring compactness. These scores are higher the
        more compact the resulting district would be if this precinct were
        added to a district.
        """
        # Instead of hardcoding the method, it could be supplied as a KWarg

        if self.compactness_method == 'CSPC':
            self.G_ = self.CSPC_score(dstrct)

        elif self.compactness_method == 'circular':
            self.G_ = self.circular_score(dstrct)

        elif self.compactness_method == 'convexity':
            self.G_ = self.convexity_score(dstrct)

        elif self.compactness_method == 'average':
            self.G_ = np.mean([self.w_R*self.circular_score(dstrct),
                               self.w_C*self.convexity_score(dstrct)])
        elif self.compactness_method == 'sum':
            self.G_ = self.w_R*self.circular_score(dstrct) + \
                      self.w_C*self.convexity_score(dstrct)
        else:
            msg = ("Valid compactness methods are "
                   "'CSPC', 'circular', 'convexity', 'average', and 'sum'")
            raise Exception(msg)

        # While all neighbors are unassigned, we want to weight compactness
        multiplier = 2 - any([pt.assigned for pt in dstrct.neighbors])
        # self.G_ *= self.w_G * (2-self.assigned)
        self.G_ *= self.w_G * multiplier
        return self.G_


    def E(self, dstrct):
        """
        Calculates a score representing how much the addition of this
        precinct would reduce the district's efficiency gap.
        """
        rep_votes = dstrct.rep_votes + self.properties['sen_red']
        dem_votes = dstrct.dem_votes + self.properties['sen_blue']

        thresh = threshold(rep_votes+dem_votes)
        rep_wasted = wasted_votes(rep_votes, thresh)
        dem_wasted = wasted_votes(dem_votes, thresh)
        gap = (rep_wasted - dem_wasted)/(rep_votes + dem_votes)
        score = 1-abs(gap)

        self.E_ = self.w_E * score
        return self.E_


    def F(self, dstrct, ):
        """Sum of heuristic functions used to direct districting"""
        # If this precinct is land locked by the same district, it should
        # automatically be absorbed. No other district could get to it.
        nbr_districts = set([pt.dstrct for pt in self.neighbors])
        if len(nbr_districts) == 1:
            self.F_ = 101 #dalmations
        else:
            self.F_ = 0
            if self.w_H:
                self.F_ += self.H(dstrct)
            if self.w_G:
                self.F_ += self.G(dstrct)
            if self.w_E:
                self.F_ += self.E(dstrct)

        return self.F_
