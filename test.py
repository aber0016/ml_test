"""
Author: Armin Berger
Date: 10.01.23

Practice Project which seeks to use Pytorch and FluidML to classify SMS text
messages as spam (1) or not (0).
"""
import pandas as pd
import os
import nltk
import numpy as np

#nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import fluidml
from fluidml.common import Task
from fluidml.swarm import Swarm
from fluidml.flow import Flow, GridTaskSpec, TaskSpec
from fluidml.visualization import visualize_graph_interactive
from fluidml.visualization import visualize_graph_in_console

