import pandas as pd
import numpy as np
from pkg_resources import add_activation_listener
import seaborn as sns
import matplotlib.pyplot as plt
import tabula as tb
import re

from tabula.io import convert_into
import pdfplumber
import camelot
from tabula import read_pdf
import tabulate

file = read_pdf("/Users/unun/Desktop/Tosang Beach Project/Day Data/31-10-21.PDF")
file

tb.convert_into("/Users/unun/Desktop/Tosang Beach Project/Day Data/31-10-21.PDF", "first_try_df.csv", output_format="csv")

file['Adt']

df = pd.read_csv("first_try_df.csv")

df['Adt']