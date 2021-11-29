from bs4 import BeautifulSoup
import requests
import lxml
import pandas as pd
import numpy as np

html_text = requests.get('http://www.columbiarecords.com/artists/').text
soup = BeautifulSoup(html_text, 'lxml')
artist_col = soup.find_all('h2')
print(artist_col)
for name in artist_col:
    print(name.text)


html_text = requests.get('https://www.rcarecords.com/artists/').text
soup = BeautifulSoup(html_text, 'lxml')
artist_rca = soup.find_all('h3')
print(artist_rca)
for name in artist_rca:
    print(name.text)

html_text = requests.get('https://www.epicrecords.com/').text
soup = BeautifulSoup(html_text, 'lxml')
artist_epic = soup.find_all('h3')
print(artist_epic)
for name in artist_epic:
    print(name.text)

html_text = requests.get('https://www.aristarecordings.com/').text
soup = BeautifulSoup(html_text, 'lxml')
artist_arista = soup.find_all('h4')
print(artist_arista)
for name in artist_arista:
    print(name.text)

html_text = requests.get('https://www.sonymusiclatin.com/artists/').text
soup = BeautifulSoup(html_text, 'lxml')
artist_latin = soup.find_all('h2')
print(artist_latin)
for name in artist_latin:
    print(name.text)

html_text = requests.get('https://www.sonymusicmasterworks.com/page/6/#artists').text
soup = BeautifulSoup(html_text, 'lxml')
artist_master = soup.find_all('a')
print(artist_master)
for name in artist_master:
    print(name.text)

html_text = requests.get('https://www.rcainspiration.com/artists/').text
soup = BeautifulSoup(html_text, 'lxml')
artist_rcainsp = soup.find_all('span')
print(artist_rcainsp)
for name in artist_rcainsp:
    print(name.text)

html_text = requests.get('https://www.providentlabelgroup.com/').text
soup = BeautifulSoup(html_text, 'lxml')
artist_prov = soup.find_all('p')
print(artist_prov)
for name in artist_prov:
    print(name.text)

html_text = requests.get('https://sonyclassical.com/artists?page=3').text
soup = BeautifulSoup(html_text, 'lxml')
artist_classical = soup.find_all('a')
print(artist_classical)
for name in artist_classical:
    print(name.text)

html_text = requests.get('https://www.legacyrecordings.com/artists/').text
soup = BeautifulSoup(html_text, 'lxml')
artist_legacy = soup.find_all('a', class_="artist-name modal-link")
print(artist_legacy)
for name in artist_legacy:
    print(name.text)

import seaborn as sns