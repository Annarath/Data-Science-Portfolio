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

31_10 = read_pdf("/Users/unun/Desktop/Tosang Beach Project/Day Data/31-10-21.PDF")
tb.convert_into("/Users/unun/Desktop/Tosang Beach Project/Day Data/31-10-21.PDF", "31_10.csv", output_format="csv")
df31_10 = pd.read_csv("31_10.csv")

30_10 = read_pdf("/Users/unun/Desktop/Tosang Beach Project/Day Data/30-10-21.PDF")
tb.convert_into("/Users/unun/Desktop/Tosang Beach Project/Day Data/30-10-21.PDF", "30_10.csv", output_format="csv")
df30_10 = pd.read_csv("30_10.csv")

29_10 = read_pdf("/Users/unun/Desktop/Tosang Beach Project/Day Data/29-10-21.PDF")
tb.convert_into("/Users/unun/Desktop/Tosang Beach Project/Day Data/29-10-21.PDF", "29_10.csv", output_format="csv")
df29_10 = pd.read_csv("29_10.csv")

28_10 = read_pdf("/Users/unun/Desktop/Tosang Beach Project/Day Data/28-10-21.PDF")
tb.convert_into("/Users/unun/Desktop/Tosang Beach Project/Day Data/28-10-21.PDF", "28_10.csv", output_format="csv")
df28_10 = pd.read_csv("28_10.csv")

27_10 = read_pdf("/Users/unun/Desktop/Tosang Beach Project/Day Data/27-10-21.PDF")
tb.convert_into("/Users/unun/Desktop/Tosang Beach Project/Day Data/27-10-21.PDF", "27_10.csv", output_format="csv")
df27_10 = pd.read_csv("27_10.csv")

26_10 = read_pdf("/Users/unun/Desktop/Tosang Beach Project/Day Data/26-10-21.PDF")
tb.convert_into("/Users/unun/Desktop/Tosang Beach Project/Day Data/26-10-21.PDF", "26_10.csv", output_format="csv")
df26_10 = pd.read_csv("26_10.csv")

25_10 = read_pdf("/Users/unun/Desktop/Tosang Beach Project/Day Data/25-10-21.PDF")
tb.convert_into("/Users/unun/Desktop/Tosang Beach Project/Day Data/25-10-21.PDF", "25_10.csv", output_format="csv")
df25_10 = pd.read_csv("25_10.csv")

24_10 = read_pdf("/Users/unun/Desktop/Tosang Beach Project/Day Data/24-10-21.PDF")
tb.convert_into("/Users/unun/Desktop/Tosang Beach Project/Day Data/24-10-21.PDF", "24_10.csv", output_format="csv")
df24_10 = pd.read_csv("24_10.csv")

23_10 = read_pdf("/Users/unun/Desktop/Tosang Beach Project/Day Data/23-10-21.PDF")
tb.convert_into("/Users/unun/Desktop/Tosang Beach Project/Day Data/23-10-21.PDF", "23_10.csv", output_format="csv")
df23_10 = pd.read_csv("23_10.csv")

22_10 = read_pdf("/Users/unun/Desktop/Tosang Beach Project/Day Data/22-10-21.PDF")
tb.convert_into("/Users/unun/Desktop/Tosang Beach Project/Day Data/22-10-21.PDF", "22_10.csv", output_format="csv")
df22_10 = pd.read_csv("22_10.csv")

21_10 = read_pdf("/Users/unun/Desktop/Tosang Beach Project/Day Data/21-10-21.PDF")
tb.convert_into("/Users/unun/Desktop/Tosang Beach Project/Day Data/21-10-21.PDF", "21_10.csv", output_format="csv")
df21_10 = pd.read_csv("21_10.csv")

20_10 = read_pdf("/Users/unun/Desktop/Tosang Beach Project/Day Data/20-10-21.PDF")
tb.convert_into("/Users/unun/Desktop/Tosang Beach Project/Day Data/20-10-21.PDF", "20_10.csv", output_format="csv")
df20_10 = pd.read_csv("20_10.csv")

19_10 = read_pdf("/Users/unun/Desktop/Tosang Beach Project/Day Data/19-10-21.PDF")
tb.convert_into("/Users/unun/Desktop/Tosang Beach Project/Day Data/19-10-21.PDF", "19_10.csv", output_format="csv")
df19_10 = pd.read_csv("19_10.csv")

18_10 = read_pdf("/Users/unun/Desktop/Tosang Beach Project/Day Data/18-10-21.PDF")
tb.convert_into("/Users/unun/Desktop/Tosang Beach Project/Day Data/18-10-21.PDF", "18_10.csv", output_format="csv")
df18_10 = pd.read_csv("18_10.csv")