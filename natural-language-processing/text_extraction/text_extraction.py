# TEXT EXTRACTION USING NATURAL LANGUAGE TOOLKIT

# import libraries
import urllib.request # url handling 
from bs4 import BeautifulSoup # handling or parsing HTML files
import nltk
nltk.download('stopwords') # download stopwords 
from nltk.corpus import stopwords

# get data from website
response = urllib.request.urlopen('https://en.wikipedia.org/wiki/spaceX') # open website url
html = response.read() 
soup = BeautifulSoup(html, 'html5lib') # specify html format
text = soup.get_text(strip = True) # get the text only

# # get data from txt file
# file = open('natural-language-processing/title_generation/paragraph.txt', 'r')
# text = file.read()

# tokenize data
tokens = [t for t in text.split()]

# get stopwords
stopword = stopwords.words('english')
clean_tokens = tokens[:]

for token in tokens:
    if token in stopwords.words('english'):
        clean_tokens.remove(token) # remove stopwords in the token
    
freq = nltk.FreqDist(clean_tokens) # frequency of keywords
for key, val in freq.items():
    print(str(key) + ':' + str(val))

# plot frequency values
freq.plot(20, cumulative = False)
