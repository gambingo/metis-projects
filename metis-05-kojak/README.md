December 2017

### Fighting Gerrymandering with Data Science
In most states voters don't pick their legislators, rather legislators pick their votes. Gerrymandering is widely viewed as poisonous to democracy. Representatives in non-competitive districts only need to compete in the primary elections, encouraging candidates with extreme views and widening the partisan gap.

Legislators often claim that since Democratic voters tend to be packed into cities and Republican voters are spread across rural towns, it would be impossible to draw competitive maps at all. They use this notion to justify their blatant partisan gerrymandering. We couldn't draw a fair map even if we tried, so we might as draw a ridiculously unfair map.

With only three weeks of hard work, I generated maps that prove their claim in nonsense. The maps the tool currently draws are not perfect, but they are still much better than the current districts. I believe that an independent redistricting committee could use algorithms such as this to help inform and guide their work.

### Data
For this project, I focussed on North Carolina. Since the North Carolina legislators especially brash gerrymandering after the most recent census, much work has been done to collect data on these districts and the precincts that make them up. All of this data is publically available but can be difficult to track down and tie together. I am very grateful to [Michal Migurski](http://mike.teczno.com/) who collected [GeoJSON and demographic data](http://mike.teczno.com/notes/redistricting/building-north-carolina-data.html) for all electoral precincts in North Carolina. Right now, my code expects data in his format.

### The Algorithm
When looking at the map of North Carolina electoral precincts, I realized I needed to find an optimal way to cluster them together. In research on what clustering algorithm would be best for drawing maps, I came across the following algorithm which has been designed specifically for redistricting.

In their paper [Redistricting using Heuristic-Based Polygonal Clustering](http://cse.unl.edu/~lksoh/pubs/conference/Joshietal_ICDM2009.pdf), Deepti Joshi, Leen-Kiat Soh, and Ashok Samal define a technique for using heuristic equations to guide which precincts should be added to which districts. In their algorithm, they only used heuristic functions that controlled population and compactness. I saw an opportunity to define a heuristic function for election competitiveness.

##### Compactness Metrics
* [Circularity](http://cho.pol.illinois.edu/wendy/papers/talismanic.pdf)
* [Convexity](http://www.bmva.org/bmvc/2002/papers/61/full_61.pdf)

##### Election Competiveness
* [The Efficiency Gap](https://www.brennancenter.org/sites/default/files/legal-work/How_the_Efficiency_Gap_Standard_Works.pdf)

### Code
##### Python Scripts
1. `redistricting.py`
    Redistricting class: load state data, define heuristic weights, grow districts.
2. `district.py`
    Sub-class used by `redistricting.py` to track districts as they grow and calculate district-level heuristic function scores.
3. `precinct.py`
    Sub-class used by `redistricting.py` to track precincts and calculate precinct-level heuristic function scores.
4. `redplotlib.py`
    Redistricting Plotting Library.
5. `kojak.py`
    Various helper functions shared across classes.
6. `run_on_AWS.py`
    As the name suggests, this script loads all data 

##### Jupyter Notebooks
1. `redistricting.ipynb`
    A demonstration of the algorithm on a small subset of North Carolina. 
2. `dev-efficiency-gap.ipynb`
    Demonstrating how to calculate the efficiency gap for North Carolina.

##### Bash Scripts
1. `bootstrap.sh`
    After cloning this repo, run `source bash_scripts/bootstrap.sh` to install MiniConda and install all packages needed to run the code.
