import pandas as pd
import os, re
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from bokeh.io import output_file, show, save
from bokeh.layouts import gridplot
from bokeh.palettes import Viridis3
from bokeh.plotting import figure
from bokeh.palettes import Spectral11
from bokeh.models import Legend, Range1d
from bokeh.plotting import figure, show, output_file
from datetime import datetime


def timeseries_plot(df, plot_param, time_param, filename=None):
	if not filename:
		filename = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	df_plot = df[plot_param+[time_param]]
	df_plot[time_param] = pd.to_datetime(df_plot[time_param], errors='coerce')
	df_plot = df_plot[df_plot[time_param].notnull()]
	df_plot.index = pd.DatetimeIndex(df_plot[time_param])
	p = figure(width=700, height=300, x_axis_type="datetime")
	line_colors = ["blue", "orange", "green", "purple"]
	for i in range(len(plot_param)):
	    r = p.line(df_plot.index.values, df_plot[plot_param[i]], color=line_colors[i])
	output_file(filename + '.html')
	save(p)