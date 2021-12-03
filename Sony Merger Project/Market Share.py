import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dfms = pd.read_csv("/Users/unun/Desktop/Sony Project/UK revenue 2015-2020.csv")
dfms

#calculate HHI

dfms['Sony'].plot(kind='hist', bins=30)