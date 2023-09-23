#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


comments = pd.read_csv(r'C:/Users/91886/Desktop/Blackcofer/Input.csv', error_bad_lines=False)


# In[ ]:





# In[ ]:





# In[8]:


comments.dropna(inplace=True)


# In[9]:


comments.isnull().sum()


# In[ ]:





# In[10]:


comments.isnull()


# In[11]:


get_ipython().system('pip install textblob')


# In[12]:


from textblob import TextBlob


# In[24]:


pip install requests beautifulsoup4 pandas matplotlib


# In[25]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt


# # Data Extraction Starts

# In[1]:


import pandas as pd
import requests
from bs4 import BeautifulSoup

df = pd.read_csv(r'C:/Users/91886/Desktop/Blackcofer/Input.csv')

titles = []
articles = []

for url in df['URL']:
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for any errors in the request

        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.title.string.strip() if soup.title else "No title found"

        article = soup.find('div', class_='td-post-content tagdiv-type').get_text()  # Replace with your own logic

        titles.append(title)
        articles.append(article)

    except Exception as e:
        print(f"Error processing {url}: {str(e)}")

result_df = pd.DataFrame({'Title': titles, 'Article': articles})

result_df.to_csv('extracted_data.csv', index=False)

result_df.head()


# In[4]:


result_df.head()


# In[ ]:





# In[7]:


get_ipython().system('pip install textblob')


# In[34]:


#Positive Score


# In[8]:


from textblob import TextBlob


title_positive_scores = []
article_positive_scores = []

for index, row in result_df.iterrows():
    title_blob = TextBlob(row['Title'])
    article_blob = TextBlob(row['Article'])
    
    title_sentiment = title_blob.sentiment.polarity
    article_sentiment = article_blob.sentiment.polarity
    
    title_positive_scores.append(title_sentiment)
    article_positive_scores.append(article_sentiment)

result_df['Title_Positive_Score'] = title_positive_scores
result_df['Article_Positive_Score'] = article_positive_scores

result_df.head()


# In[35]:


#Negative Score


# In[10]:


from textblob import TextBlob


title_negative_scores = []
article_negative_scores = []

for index, row in result_df.iterrows():
    title_blob = TextBlob(row['Title'])
    article_blob = TextBlob(row['Article'])
    
    title_sentiment = title_blob.sentiment.polarity
    article_sentiment = article_blob.sentiment.polarity
    
    title_negative_scores.append(-title_sentiment)
    article_negative_scores.append(-article_sentiment)

result_df['Title_Negative_Score'] = title_negative_scores
result_df['Article_Negative_Score'] = article_negative_scores

result_df.head()


# In[36]:


#Polarity Score


# In[12]:


from textblob import TextBlob


title_polarity_scores = []
article_polarity_scores = []

for index, row in result_df.iterrows():
    title_blob = TextBlob(row['Title'])
    article_blob = TextBlob(row['Article'])
    
    title_polarity = title_blob.sentiment.polarity
    article_polarity = article_blob.sentiment.polarity
    
    title_polarity_scores.append(title_polarity)
    article_polarity_scores.append(article_polarity)

result_df['Title_Polarity_Score'] = title_polarity_scores
result_df['Article_Polarity_Score'] = article_polarity_scores

result_df.head()


# In[13]:


#Subjective Score


# In[14]:


from textblob import TextBlob


title_subjectivity_scores = []
article_subjectivity_scores = []

for index, row in result_df.iterrows():
    title_blob = TextBlob(row['Title'])
    article_blob = TextBlob(row['Article'])
    
    title_subjectivity = title_blob.sentiment.subjectivity
    article_subjectivity = article_blob.sentiment.subjectivity
    
    title_subjectivity_scores.append(title_subjectivity)
    article_subjectivity_scores.append(article_subjectivity)

result_df['Title_Subjectivity_Score'] = title_subjectivity_scores
result_df['Article_Subjectivity_Score'] = article_subjectivity_scores

result_df.head()


# In[ ]:





# In[ ]:





# In[15]:


#Average Sentence Length


# In[16]:


get_ipython().system('pip install nltk')


# In[17]:


import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize


title_avg_sentence_lengths = []
article_avg_sentence_lengths = []

for index, row in result_df.iterrows():
    title_sentences = sent_tokenize(row['Title'])
    article_sentences = sent_tokenize(row['Article'])
    
    title_avg_sentence_length = sum(len(sentence.split()) for sentence in title_sentences) / len(title_sentences) if title_sentences else 0
    article_avg_sentence_length = sum(len(sentence.split()) for sentence in article_sentences) / len(article_sentences) if article_sentences else 0
    
    title_avg_sentence_lengths.append(title_avg_sentence_length)
    article_avg_sentence_lengths.append(article_avg_sentence_length)

result_df['Title_Avg_Sentence_Length'] = title_avg_sentence_lengths
result_df['Article_Avg_Sentence_Length'] = article_avg_sentence_lengths

result_df.head()


# In[ ]:





# In[18]:


#Percentage of the Complex Words


# In[19]:


import nltk
nltk.download('punkt')
nltk.download('cmudict')

from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict

cmu_dict = cmudict.dict()

def count_syllables(word):
    if word.lower() in cmu_dict:
        return max([len(list(y for y in x if y[-1].isdigit())) for x in cmu_dict[word.lower()]])
    else:
        return 1

def calculate_complexity_percentage(text):
    words = word_tokenize(text)
    complex_word_count = sum(1 for word in words if count_syllables(word) >= 3)
    total_word_count = len(words)
    return (complex_word_count / total_word_count) * 100 if total_word_count > 0 else 0


title_complexity_percentages = []
article_complexity_percentages = []

for index, row in result_df.iterrows():
    title_complexity = calculate_complexity_percentage(row['Title'])
    article_complexity = calculate_complexity_percentage(row['Article'])
    
    title_complexity_percentages.append(title_complexity)
    article_complexity_percentages.append(article_complexity)

result_df['Title_Complexity_Percentage'] = title_complexity_percentages
result_df['Article_Complexity_Percentage'] = article_complexity_percentages

result_df.head()


# In[ ]:





# In[ ]:





# In[20]:


import nltk
nltk.download('punkt')
nltk.download('cmudict')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import cmudict

cmu_dict = cmudict.dict()

def count_syllables(word):
    if word.lower() in cmu_dict:
        return max([len(list(y for y in x if y[-1].isdigit())) for x in cmu_dict[word.lower()]])
    else:
        return 1

def calculate_fog_index(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    total_sentences = len(sentences)
    total_words = len(words)
    
    average_sentence_length = total_words / total_sentences if total_sentences > 0 else 0
    
    complex_word_count = sum(1 for word in words if count_syllables(word) >= 3)
    complex_word_percentage = (complex_word_count / total_words) * 100 if total_words > 0 else 0
    
    fog_index = 0.4 * (average_sentence_length + complex_word_percentage)
    
    return fog_index


title_fog_scores = []
article_fog_scores = []

for index, row in result_df.iterrows():
    title_fog = calculate_fog_index(row['Title'])
    article_fog = calculate_fog_index(row['Article'])
    
    title_fog_scores.append(title_fog)
    article_fog_scores.append(article_fog)

result_df['Title_FOG_Index'] = title_fog_scores
result_df['Article_FOG_Index'] = article_fog_scores

result_df.head()


# In[ ]:





# In[ ]:





# In[21]:


#Average number of the words per sentence


# In[22]:


import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize

def calculate_avg_words_per_sentence(text):
    sentences = sent_tokenize(text)
    total_sentences = len(sentences)
    
    if total_sentences == 0:
        return 0  # Avoid division by zero
    
    total_words = sum(len(word_tokenize(sentence)) for sentence in sentences)
    avg_words_per_sentence = total_words / total_sentences
    
    return avg_words_per_sentence


title_avg_words_per_sentence = []
article_avg_words_per_sentence = []

for index, row in result_df.iterrows():
    title_avg_words = calculate_avg_words_per_sentence(row['Title'])
    article_avg_words = calculate_avg_words_per_sentence(row['Article'])
    
    title_avg_words_per_sentence.append(title_avg_words)
    article_avg_words_per_sentence.append(article_avg_words)

result_df['Title_Avg_Words_Per_Sentence'] = title_avg_words_per_sentence
result_df['Article_Avg_Words_Per_Sentence'] = article_avg_words_per_sentence

result_df.head()


# In[ ]:





# In[23]:


#Complex Word count


# In[24]:


import nltk
nltk.download('punkt')
nltk.download('cmudict')

from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict

cmu_dict = cmudict.dict()

def count_syllables(word):
    if word.lower() in cmu_dict:
        return max([len(list(y for y in x if y[-1].isdigit())) for x in cmu_dict[word.lower()]])
    else:
        return 1

def calculate_complex_word_count(text):
    words = word_tokenize(text)
    complex_word_count = sum(1 for word in words if count_syllables(word) >= 3)
    return complex_word_count


title_complex_word_counts = []
article_complex_word_counts = []

for index, row in result_df.iterrows():
    title_complex_count = calculate_complex_word_count(row['Title'])
    article_complex_count = calculate_complex_word_count(row['Article'])
    
    title_complex_word_counts.append(title_complex_count)
    article_complex_word_counts.append(article_complex_count)

result_df['Title_Complex_Word_Count'] = title_complex_word_counts
result_df['Article_Complex_Word_Count'] = article_complex_word_counts

result_df.head()


# In[ ]:





# In[ ]:





# In[25]:


#word count calculate


# In[26]:


from nltk.tokenize import word_tokenize

def calculate_word_count(text):
    words = word_tokenize(text)
    return len(words)

title_word_counts = []
article_word_counts = []

for index, row in result_df.iterrows():
    title_word_count = calculate_word_count(row['Title'])
    article_word_count = calculate_word_count(row['Article'])
    
    title_word_counts.append(title_word_count)
    article_word_counts.append(article_word_count)

result_df['Title_Word_Count'] = title_word_counts
result_df['Article_Word_Count'] = article_word_counts

result_df.head()


# In[28]:


#SYLLABLE PER WORD


# In[27]:


import nltk
nltk.download('punkt')
nltk.download('cmudict')

from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict

cmu_dict = cmudict.dict()

def count_syllables(word):
    if word.lower() in cmu_dict:
        return max([len(list(y for y in x if y[-1].isdigit())) for x in cmu_dict[word.lower()]])
    else:
        return 1

def calculate_avg_syllables_per_word(text):
    words = word_tokenize(text)
    total_syllables = sum(count_syllables(word) for word in words)
    total_words = len(words)
    
    if total_words == 0:
        return 0  # Avoid division by zero
    
    avg_syllables_per_word = total_syllables / total_words
    
    return avg_syllables_per_word

title_avg_syllables_per_word = []
article_avg_syllables_per_word = []

for index, row in result_df.iterrows():
    title_avg_syllables = calculate_avg_syllables_per_word(row['Title'])
    article_avg_syllables = calculate_avg_syllables_per_word(row['Article'])
    
    title_avg_syllables_per_word.append(title_avg_syllables)
    article_avg_syllables_per_word.append(article_avg_syllables)

result_df['Title_Avg_Syllables_Per_Word'] = title_avg_syllables_per_word
result_df['Article_Avg_Syllables_Per_Word'] = article_avg_syllables_per_word

result_df.head()


# In[29]:


#PERSONAL PRONOUNS


# In[30]:


import pandas as pd
import re

data = {'Title': ["I love programming.", "He enjoys reading.", "She is a scientist."],
        'Article': ["They visited the museum.", "We went to the beach.", "You should try this recipe."]}
df = pd.DataFrame(data)

def calculate_personal_pronoun_frequency(text):
    pronoun_pattern = r'\b(I|me|my|mine|myself|you|your|yours|yourself|he|him|his|himself|she|her|hers|herself|it|its|itself|we|us|our|ours|ourselves|they|them|their|theirs|themselves)\b'
    pronouns = re.findall(pronoun_pattern, text, re.IGNORECASE)
    return len(pronouns)

df['Title_Pronoun_Frequency'] = df['Title'].apply(calculate_personal_pronoun_frequency)
df['Article_Pronoun_Frequency'] = df['Article'].apply(calculate_personal_pronoun_frequency)

print(df)


# In[31]:


import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize

def calculate_avg_word_length(text):
    words = word_tokenize(text)
    total_characters = sum(len(word) for word in words)
    total_words = len(words)
    
    if total_words == 0:
        return 0  # Avoid division by zero
    
    avg_word_length = total_characters / total_words
    
    return avg_word_length


title_avg_word_lengths = []
article_avg_word_lengths = []

for index, row in result_df.iterrows():
    title_avg_length = calculate_avg_word_length(row['Title'])
    article_avg_length = calculate_avg_word_length(row['Article'])
    
    title_avg_word_lengths.append(title_avg_length)
    article_avg_word_lengths.append(article_avg_length)

result_df['Title_Avg_Word_Length'] = title_avg_word_lengths
result_df['Article_Avg_Word_Length'] = article_avg_word_lengths

result_df.head()


# In[ ]:





# In[ ]:





# In[32]:


#create a file


# In[33]:


import pandas as pd

# Assuming you have a DataFrame called result_df with all the calculated values and original data

# Define the file name for the output CSV file
output_file_name = 'Output_Data_Structure.csv'

# Save the DataFrame to a new CSV file
result_df.to_csv(output_file_name, index=False)

# Confirm that the file has been saved
print(f"The data has been saved to {output_file_name}")


# In[ ]:




