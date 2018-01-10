import numpy as np
from math import pi
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.geometry import Polygon

# import sys
# path = '/Users/Joe/Documents/Metis/Projects/metis-05-kojak/python-scripts/'
# sys.path.append(path)
from kojak import threshold
from kojak import wasted_votes


class district:

    def __init__(self, weights, compactness_method, method_weights, fc):
        """
        Stores information for each district (district).
        Args:
            target_pop: Target population for each district.
                        State population divided by k.
        """
        self.x = None #target population
        self.pop_proportion = None
        self.current_pop = 0
        self.rep_votes = 0
        self.dem_votes = 0
        self.regenerate_neighbors = True
        self.fc = fc

        self.members = []
        self.geometry = Polygon()
        self.centroid = None
        self.neighbors = []

        # Heuristic functions
        self.H_ = 1.0
        self.gap = 1.0
        self.F_ = None
        # Weights for each heuristic function
        self.w_H = weights[0] #population
        self.w_G = weights[1] #compactness
        self.w_E = weights[2] #efficiency gap

        self.compactness_method = compactness_method
        self.w_R = method_weights[0]
        self.w_C = method_weights[1]


    def add_member(self, pt):
        """
        Add a precinct to the district and adjust attributes accordingly
        """
        # From the first member, grab the target population and centroid
        if not self.x:
            self.x = pt.x

        if not self.centroid:
            self.centroid = pt.centroid

        # Adjust the precinct's attributes
        pt.assigned = True
        self.regenerate_neighbors = True

        # Must recalculate F function
        self.current_pop += pt.properties['population']
        self.rep_votes += pt.properties['sen_red']
        self.dem_votes += pt.properties['sen_blue']
        self.F_ = None

        self.members.append(pt)
        self.geometry = unary_union([self.geometry, pt.geometry])


    def remove_member(self, pt):
        """
        Remove a precinct to the district and adjust attributes accordingly
        """
        pt.assigned = False
        ii = self.members.index(pt)
        self.members = self.members[0:ii] + self.members[ii+1:]

        self.current_pop -= pt.properties['population']
        self.rep_votes -= pt.properties['sen_red']
        self.dem_votes -= pt.properties['sen_blue']
        self.geometry = self.geometry.difference(pt.geometry)
        self.regenerate_neighbors = True
        self.F_ = None


    def H(self):
        """
        Heuristic function controlling population of a district.
        ---
        This is normalized so an empty district has a score of 1 and a full
        district has score of one.
        """
        self.H_ = (self.x - self.current_pop)/self.x
        # If you take the absolute value of this score, and one district gets
        # too large, it will run away and consume the whole state.
        # score = abs(score)
        return self.w_H * self.H_


    def CSPC_score(self):
        """
        Compactness Method 1:
        The CSPC specifies the following method. The seems to work well for
        regular shaped self.precincts like those of Nebraska, but it doesn't
        generalize well to the irregular shaped self.precincts of North
        Carolina.

        For each (unassigned) neighboring polygon, calculate:
            (overlap + new_border)/overlap
        where:
            overlap =   length of district border shared with unassigned
                        neighbors
            new_border= length of the district border if this polygon were
                        added
        """
        g = []
        overlap = 0
        for pt in self.neighbors:
            overlap += self.geometry.intersection(pt.geometry).length

        for pt in self.neighbors:
            candidate_dstrct = unary_union([self.geometry, pt.geometry])
            new_border = candidate_dstrct.length
            g.append((overlap+new_border)/overlap)

        return max(g)


    def circular_score(self):
        """
        Compactness Method 2:
        A second method is to calculate the ratio of border length to
        polygon area. The ratio is normalized to returns 1 for a circle.
        ---
        The ratio is higher the more compact a district.
        The score must be higher the less compact a district.
        This score is weighted by how many members are in the district
        because a district with more members will be less compact.
        """
        # geom = self.geometry.buffer(0.001)
        # ratio = (geom.area*4*pi)/(geom.length**2)
        ratio = (self.geometry.area*4*pi)/(self.geometry.length**2)
        score = 1-ratio
        # size_factor = len(self.members)/10
        # return score/size_factor
        return score


    def convexity_score(self):
        """
        Compactness Method 3:
        The convexity ratio measures the ratio a shapes area to the area
        of the smallest convex shape that contains all points in the
        shape. It equals one for a fully convex shape.
        ---
        The ratio is higher the more convex a shape.
        This score must be higher the less convex a shape.
        """
        A_shape = self.geometry.area
        A_hull  = self.geometry.convex_hull.area
        convexity = A_shape/A_hull
        score = 1-convexity
        # size_factor = len(self.members)/10
        # return score/size_factor
        return score


    def G(self):
        """
        Heuristic function controlling compactness of a district. There are
        several methods for scoring compactness.
        """
        if self.compactness_method == 'CSPC':
            self.G_ = self.w_G * self.CSPC_score()

        elif self.compactness_method == 'circular':
            self.G_ = self.w_G * self.circular_score()

        elif self.compactness_method == 'convexity':
            self.G_ = self.w_G * self.convexity_score()

        elif self.compactness_method == 'average':
            # print('Ratio Score: ', self.circular_score())
            # print('Convexity Score: ', self.convexity_score())
            self.G_ = self.w_G * np.mean([self.w_R * self.circular_score(),
                                          self.w_C * self.convexity_score()])
        elif self.compactness_method == 'sum':
            self.G_ = self.w_R*self.circular_score() + \
                      self.w_C*self.convexity_score()
        else:
            msg = ("Valid compactness methods are "
                   "'CSPC', 'circular', 'convexity', 'average', and 'sum'")
            raise Exception(msg)

        return self.G_/len(self.members)
        # return self.G_


    def E(self):
        """
        Calculates the Efficiency Gap for the district in its current state.
        """
        thresh = threshold(self.rep_votes + self.dem_votes)
        rep_wasted = wasted_votes(self.rep_votes, thresh)
        dem_wasted = wasted_votes(self.dem_votes, thresh)

        self.gap = (rep_wasted - dem_wasted)/(self.rep_votes + self.dem_votes)
        self.gap = abs(self.gap)
        self.E_ = self.w_E * self.gap
        return self.E_


    def F(self):
        """Sum of heuristic functions used to direct districting"""
        if self.F_:
            return self.F_
        else:
            self.F_ = 1
            if self.w_H:
                self.F_ *= self.H()
            if self.w_G:
                self.F_ *= self.G()
            if self.w_E:
                self.F_ *= self.E()

            # self.F_ = self.F_/sum([self.w_H, self.w_G, self.w_E])
            return self.F_
