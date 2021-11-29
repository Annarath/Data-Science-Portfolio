import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tabula as tb
import re

room = '01-10-21.pdf'
data = tb.read_pdf(room)
