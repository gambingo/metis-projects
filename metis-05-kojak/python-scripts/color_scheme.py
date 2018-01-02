"""
Defines various colors used by different plotting functions.
"""
from random import shuffle

colors = {'green':      '#45735F',
          'light green':'#A6BF80',
          'crimson':    '#DC143C',
          'soft red':   '#BE4248',
          'royal blue': '#4169E1',
          'soft blue':  '#5AA6CF',
          'dark blue':  '#35478C',
          'dark purple':'#36175E',
          'blend purp': '#35478C',
          'white':      '#FFFFFF',
          'gray':       '#999999',
          'soft gray':  '#586473',
          'pale gray':  '#E7DACB',
          'black':      '#000000'}

scheme = {'earth':      colors['pale gray'],
          'rep':        colors['soft red'],
          'dem':        colors['dark blue'],
          'uncontested':colors['dark purple'],
          'pt border':  colors['soft gray'],
          'dst border': colors['soft gray']}

lw_pt = 0.1
lw_dstrct = 0.5

palette =['#56B9D0', '#FEFEFE', '#FBBA42', '#F24C27', '#3B3F42', '#F28C8C',
          scheme['rep'], scheme['uncontested']]
shuffle(palette)
palette = iter(palette + palette)
