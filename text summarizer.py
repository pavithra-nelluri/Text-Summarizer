import nltk
nltk.download('wordnet')
import math
from nltk import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from itertools import product

def preprocess_text(text):
    # Tokenize text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.lower() not in stop_words]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return sentences, words
# Example usage:

text = """
India is one of the strongest countries in the world. It is the seventh-largest country in the world area-wise and the second-most populous country in the world. India shares its borders with countries like Pakistan, Afghanistan, China, Nepal, Bhutan, Myanmar and Bangladesh. It also shares its borders with Sri Lanka and the Maldives. It is a secular and democratic country that respects all religions and the people of India have the liberty to practise any religion they want.

India believes in nonviolence and therefore, Mahatma Gandhi is known as the father of the Nation because of his non-violent contribution to the freedom movement of the country. The tri-coloured national flag is known as Tiranga which has the Ashoka Chakra at the centre. The national emblem of the country is the ‘Lion Capital of Ashoka’.

The Param Vir Chakra is India’s highest military decoration, given to those who have shown courage. Soldiers who put their lives on the line to safeguard residents are India’s heroes. Pandit Jawaharlal Nehru, often known as Pandit Nehru or Chacha Nehru, was India’s first prime minister. India is a land of many festivals, different dressing styles, and different food styles. People of different castes, creeds, and colours also live peacefully in India, and this is how it sets a perfect example of ‘Unity in Diversity’.
"""

# Remove space tabs and replace them with a single space
text = text.replace('\t', ' ')

# Remove any extra spaces
text = ' '.join(text.split())

print(text)



sentences, words = preprocess_text(text)
def calculate_title_feature_score(sentence, title):
    # Tokenize sentence and title into words
    sentence_words = word_tokenize(sentence)
    title_words = word_tokenize(title)
    
    # Calculate the number of words in the sentence that occur in the title
    matches = sum(word in title_words for word in sentence_words)
    
    # Calculate the score
    score = matches / len(title_words)
    return score

def calculate_sentence_length(sentence):
    # Tokenize sentence into words
    words = word_tokenize(sentence)
    
    # Calculate the number of words in the sentence
    length = len(words)
    
    return length


def calculate_sentence_ratio(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)  # Split text into sentences
    sentence_ratios = []

    max_length = max(len(sentence.split()) for sentence in sentences)  # Find the length of the longest sentence

    for sentence in sentences:
        words = sentence.split()
        sentence_length = len(words)
        sentence_ratio = sentence_length / max_length
        sentence_ratios.append(sentence_ratio)
    
    print("Sentence ratios:", sentence_ratios)
    return sentence_ratios


calculate_sentence_ratio(text)



 
def calculate_term_weight(sentence, words):
    # Calculate the term weight for each word in the sentence
    term_weight = sum(words.count(word) / len(words) for word in set(words))
    return term_weight

def calculate_sentence_position(sentence_index, total_sentences):
    # Calculate the sentence position score based on its index
    position_score = 1 - (sentence_index / total_sentences)
    
    return position_score

def calculate_sentence_similarity(sentence1, sentence2):
    # Calculate the sentence-to-sentence similarity using TF-IDF vectors and cosine similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    
    return similarity_score
def calculate_numerical_score_ratio(sentence):
    # Count the number of numerical data in the sentence
    numerical_count = len(re.findall(r'\d+', sentence))
    
    # Calculate the ratio of numerical data count over sentence length
    num_words = len(sentence.split())
    ratio = numerical_count / num_words
    return ratio

# # Example usage
# sentence = "The document contains 10 pages and 20 images."
# score = calculate_numerical_score_ratio(sentence)

# print(score)

def calculate_thematic_word_feature_score(sentence, thematic_words):
    # Calculate the thematic word feature score for a sentence based on thematic words
    # Note: This is a placeholder and needs customization based on your specific resources or requirements
    
    # Count the number of thematic words in the sentence
    thematic_word_count = sum(word in sentence for word in thematic_words)
    
    # Calculate the feature score
    score = thematic_word_count / len(thematic_words)
    return score


sentences, words = preprocess_text(text)
title = "India"
thematic_words = [" India"]

title_feature_scores = []
sentence_lengths = []
term_weights = []
sentence_positions = []
sentence_similarities = []
numerical_score_ratios = []
thematic_word_feature_scores = []

for index, sentence in enumerate(sentences):
    # Calculate title feature score
    title_feature_score = calculate_title_feature_score(sentence, title)
    title_feature_scores.append(title_feature_score)
    
    # Calculate sentence length
    sentence_length = calculate_sentence_length(sentence)
    sentence_lengths.append(sentence_length)
    
    # Calculate term weight
    term_weight = calculate_term_weight(sentence, words)
    term_weights.append(term_weight)
    
    # Calculate sentence position
    sentence_position = calculate_sentence_position(index, len(sentences))
    sentence_positions.append(sentence_position)
    
    # Calculate sentence similarity with the previous sentence
    if index > 0:
        sentence_similarity = calculate_sentence_similarity(sentence, sentences[index - 1])
    else:
        sentence_similarity = 0.0
    sentence_similarities.append(sentence_similarity)
    
    # Calculate numerical score ratio
    numerical_score_ratio = calculate_numerical_score_ratio(sentence)
    numerical_score_ratios.append(numerical_score_ratio)
    
    # Calculate thematic word feature score
    thematic_word_feature_score = calculate_thematic_word_feature_score(sentence, thematic_words)
    thematic_word_feature_scores.append(thematic_word_feature_score)

print("Title feature scores:", title_feature_scores)
print("Sentence lengths:", sentence_lengths)
print("Term weights:", term_weights)
print("Sentence positions:", sentence_positions)
print("Sentence similarities:", sentence_similarities)
print("Numerical score ratios:", numerical_score_ratios)
print("Thematic word feature scores:", thematic_word_feature_scores)

def _create_frequency_table(text) -> dict:
    """
    we create a dictionary for the word frequency table.
    For this, we should only use the words that are not part of the stopWords array.

    Removing stop words and making frequency table
    Stemmer - an algorithm to bring words to its root word.
    :rtype: dict
    """
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


def _create_frequency_matrix(sentences1):
    frequency_matrix = {}
    stopWords1 = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in sentences1:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords1:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix


def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix


def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table


def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


def _score_sentences(tf_idf_matrix) :
    """
    score a sentence by its word's TF
    Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = {}
    sentenceWeight = []
    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score
        
        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence
        sentenceWeight.append(sentenceValue[sent])
        
        #print("Sentence Value:", sentenceValue[sent])

    return sentenceWeight


# def _find_average_score(sentenceValue) -> int:
#     """
#     Find the average score from the sentence value dictionary
#     :rtype: int
#     """
#     sumValues = 0
#     for entry in sentenceValue:
#         sumValues += sentenceValue[entry]

#     # Average value of a sentence from original summary_text
#     average = (sumValues / len(sentenceValue))

#     return average


# def _generate_summary(sentences, sentenceValue, threshold):
#     sentence_count = 0
#     summary = ''

#     for sentence in sentences:
#         if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
#             summary += " " + sentence
#             sentence_count += 1

#     return summary


def run_summarization(text):
    """
    :param text: Plain summary_text of long article
    :return: summarized summary_text
    """

    '''
    We already have a sentence tokenizer, so we just need 
    to run the sent_tokenize() method to create the array of sentences.
    '''
    # 1 Sentence Tokenize
    sentences1 = sent_tokenize(text)
    total_documents = len(sentences1)
    #print(sentences)

    # 2 Create the Frequency matrix of the words in each sentence.
    freq_matrix = _create_frequency_matrix(sentences1)
    #print(freq_matrix)

    '''
    Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
    '''
    # 3 Calculate TermFrequency and generate a matrix
    tf_matrix = _create_tf_matrix(freq_matrix)
    #print(tf_matrix)

    # 4 creating table for documents per words
    count_doc_per_words = _create_documents_per_words(freq_matrix)
    #print(count_doc_per_words)

    '''
    Inverse document frequency (IDF) is how unique or rare a word is.
    '''
    # 5 Calculate IDF and generate a matrix
    idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
    #print(idf_matrix)

    # 6 Calculate TF-IDF and generate a matrix
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
    #print(tf_idf_matrix)

    # 7 Important Algorithm: score the sentences
    sentence_weight1 = _score_sentences(tf_idf_matrix)

    # 8 Find the threshold
    # threshold = _find_average_score(sentence_scores)
    # #print(threshold)

    # # 9 Important Algorithm: Generate the summary
    # summary = _generate_summary(sentences, sentence_scores, 1.3 * threshold)
    # return summary
    return sentence_weight1

sentence_weights2 = run_summarization(text)
print("sentence weights",sentence_weights2)
    # print("abcd",SentenceWeight1_)
#     print(result)



# title_feature_scores = []
# sentence_lengths = []
# term_weights = []
# sentence_positions = []
# sentence_similarities = []
# numerical_score_ratios = []
# thematic_word_feature_scores = []
# sentence_weights2 = []

print('-----------------------------------------------------')
# Membership function calculation for each feature
#  1. Title Feature 
def calculate_title_feature_score_membership(title_feature_scores):
    # Define fuzzy set boundaries
    long_boundary = [0.5, 1]
    short_boundary = [0, 0.5]
    middle_boundary =  [0.3, 0.8]
    # Define membership values for each fuzzy set 
    High = max(0, min((title_feature_scores - long_boundary[0]) / (long_boundary[1] - long_boundary[0]), 1))
    Low = max(0, min((short_boundary[1] - title_feature_scores) / (short_boundary[1] - short_boundary[0]), 1))
    Medium = max(0, min(1 - abs(title_feature_scores - middle_boundary[0]) / (middle_boundary[1] - middle_boundary[0]), 1))
    return {'High':High, 'Low':Low, 'Medium':Medium}
 # We need to iterate this function for every sentence.
 # Set the long,short,middle boundary

title_feature_score_dict = {}
for index, title in enumerate(title_feature_scores):
    # index1 = title_feature_scores.index(title)
    #print(index)

    title_feature_score_dict[f'sentence {index+1}']=calculate_title_feature_score_membership(title)
print("Title feature dict: ", title_feature_score_dict)

print('-----------------------------------------------------')
# 2. Sentence Length
def calculate_sentence_lengths_membership(sentence_lengths):
    # Define fuzzy set boundaries
    long_boundary = [0.5, 1]
    short_boundary = [0, 0.5]
    middle_boundary =  [0.3, 0.8]
    # Define membership values for each fuzzy set 
    Long = max(0, min((sentence_lengths - long_boundary[0]) / (long_boundary[1] - long_boundary[0]), 1))
    Short = max(0, min((short_boundary[1] - sentence_lengths) / (short_boundary[1] - short_boundary[0]), 1))
    Medium = max(0, min(1 - abs(sentence_lengths - middle_boundary[0]) / (middle_boundary[1] - middle_boundary[0]), 1))
    return {'Long':Long, 'Short':Short, 'Medium':Medium}
 # We need to iterate this function for every sentence.
 # Set the long,short,middle boundary

sentence_lengths_dict = {}
for index, title in enumerate(sentence_lengths):
    # index1 = title_feature_scores.index(title)
    #print(index)

    sentence_lengths_dict[f'sentence {index+1}']=calculate_sentence_lengths_membership(title)
print("sentence_lengths dict: ", sentence_lengths_dict)

print('-----------------------------------------------------')
# 3. Term Weights
def calculate_term_weights_membership(term_weights):
    # Define fuzzy set boundaries
    long_boundary = [0.5, 1]
    short_boundary = [0, 0.5]
    middle_boundary =  [0.3, 0.8]
    # Define membership values for each fuzzy set 
    High = max(0, min((term_weights - long_boundary[0]) / (long_boundary[1] - long_boundary[0]), 1))
    Low = max(0, min((short_boundary[1] - term_weights) / (short_boundary[1] - short_boundary[0]), 1))
    Medium = max(0, min(1 - abs(term_weights - middle_boundary[0]) / (middle_boundary[1] - middle_boundary[0]), 1))
    return {'High':High, 'Low':Low, 'Medium':Medium}
 # We need to iterate this function for every sentence.
 # Set the long,short,middle boundary

term_weights_dict = {}
for index, title in enumerate(term_weights):
    # index1 = title_feature_scores.index(title)
    #print(index)

    term_weights_dict[f'sentence {index+1}']=calculate_term_weights_membership(title)
print("term_weights dict: ", term_weights_dict)

print('-----------------------------------------------------')
# 4. sentence_positions
def calculate_sentence_positions_membership(sentence_positions):
    # Define fuzzy set boundaries
    long_boundary = [0.5, 1]
    short_boundary = [0, 0.5]
    middle_boundary =  [0.3, 0.8]
    # Define membership values for each fuzzy set 
    Start = max(0, min((sentence_positions - long_boundary[0]) / (long_boundary[1] - long_boundary[0]), 1))
    End = max(0, min((short_boundary[1] - sentence_positions) / (short_boundary[1] - short_boundary[0]), 1))
    Middle = max(0, min(1 - abs(sentence_positions - middle_boundary[0]) / (middle_boundary[1] - middle_boundary[0]), 1))
    return {'Start':Start , 'End':End, 'Middle':Middle}
 # We need to iterate this function for every sentence.
 # Set the long,short,middle boundary

sentence_positions_dict = {}
for index, title in enumerate(sentence_positions):
    # index1 = title_feature_scores.index(title)
    #print(index)

    sentence_positions_dict[f'sentence {index+1}']=calculate_sentence_positions_membership(title)
print("sentence_positions dict: ", sentence_positions_dict)

print('-----------------------------------------------------')
# 5. sentence_similarities
def calculate_sentence_similarities_membership( sentence_similarities):
    # Define fuzzy set boundaries
    long_boundary = [0.5, 1]
    short_boundary = [0, 0.5]
    middle_boundary =  [0.3, 0.8]
    # Define membership values for each fuzzy set 
    High = max(0, min(( sentence_similarities - long_boundary[0]) / (long_boundary[1] - long_boundary[0]), 1))
    Low = max(0, min((short_boundary[1] -  sentence_similarities) / (short_boundary[1] - short_boundary[0]), 1))
    Medium = max(0, min(1 - abs( sentence_similarities - middle_boundary[0]) / (middle_boundary[1] - middle_boundary[0]), 1))
    return {'High':High, 'Low':Low, 'Medium':Medium}
 # We need to iterate this function for every sentence.
 # Set the long,short,middle boundary

sentence_similarities_dict = {}
for index, title in enumerate( sentence_similarities):
    # index1 = title_feature_scores.index(title)
    #print(index)

    sentence_similarities_dict[f'sentence {index+1}']=calculate_sentence_similarities_membership(title)
print(" sentence_similarities dict: ", sentence_similarities_dict)

print('-----------------------------------------------------')
# 6. numerical_score_ratios
def calculate_numerical_score_ratios_membership(numerical_score_ratios):
    # Define fuzzy set boundaries
    long_boundary = [0.5, 1]
    short_boundary = [0, 0.5]
    middle_boundary =  [0.3, 0.8]
    # Define membership values for each fuzzy set 
    High = max(0, min(( numerical_score_ratios - long_boundary[0]) / (long_boundary[1] - long_boundary[0]), 1))
    Low = max(0, min((short_boundary[1] -  numerical_score_ratios) / (short_boundary[1] - short_boundary[0]), 1))
    Medium = max(0, min(1 - abs( numerical_score_ratios - middle_boundary[0]) / (middle_boundary[1] - middle_boundary[0]), 1))
    return {'High':High, 'Low':Low, 'Medium':Medium}
 # We need to iterate this function for every sentence.
 # Set the long,short,middle boundary

numerical_score_ratios_dict = {}
for index, title in enumerate(numerical_score_ratios):
    # index1 = title_feature_scores.index(title)
    #print(index)

    numerical_score_ratios_dict[f'sentence {index+1}'] = calculate_numerical_score_ratios_membership(title)
print(" numerical_score_ratios dict: ", numerical_score_ratios_dict)

print('-----------------------------------------------------')
# 7. thematic_word_feature_scores
def calculate_thematic_word_feature_scores_membership(thematic_word_feature_scores):
    # Define fuzzy set boundaries
    long_boundary = [0.5, 1]
    short_boundary = [0, 0.5]
    middle_boundary =  [0.3, 0.8]
    # Define membership values for each fuzzy set 
    High = max(0, min(( thematic_word_feature_scores - long_boundary[0]) / (long_boundary[1] - long_boundary[0]), 1))
    Low = max(0, min((short_boundary[1] -  thematic_word_feature_scores) / (short_boundary[1] - short_boundary[0]), 1))
    Medium = max(0, min(1 - abs(thematic_word_feature_scores - middle_boundary[0]) / (middle_boundary[1] - middle_boundary[0]), 1))
    return {'High':High, 'Low':Low, 'Medium':Medium}
 # We need to iterate this function for every sentence.
 # Set the long,short,middle boundary

thematic_word_feature_scores_dict = {}
for index, title in enumerate(thematic_word_feature_scores):
    # index1 = title_feature_scores.index(title)
    # print(index)

    thematic_word_feature_scores_dict[f'sentence {index+1}'] = calculate_thematic_word_feature_scores_membership(title)
print(" thematic_word_feature_scores dict: ",thematic_word_feature_scores_dict)

print('-----------------------------------------------------')
# 8. sentence_weights2
def calculate_sentence_weights2_membership(sentence_weights2):
    # Define fuzzy set boundaries
    long_boundary = [0.5, 1]
    short_boundary = [0, 0.5]
    middle_boundary =  [0.3, 0.8]
    # Define membership values for each fuzzy set 
    High = max(0, min((sentence_weights2 - long_boundary[0]) / (long_boundary[1] - long_boundary[0]), 1))
    Low = max(0, min((short_boundary[1] -  sentence_weights2) / (short_boundary[1] - short_boundary[0]), 1))
    Medium = max(0, min(1 - abs(sentence_weights2 - middle_boundary[0]) / (middle_boundary[1] - middle_boundary[0]), 1))
    return {'High':High, 'Low':Low, 'Medium':Medium}
 # We need to iterate this function for every sentence.
 # Set the long,short,middle boundary

sentence_weights2_dict = {}
for index, title in enumerate(sentence_weights2):
    # index1 = title_feature_scores.index(title)
    # print(index)

    sentence_weights2_dict[f'sentence {index+1}'] = calculate_sentence_weights2_membership(title)
print(" sentence_weights2 dict: ",sentence_weights2_dict)

# title_feature_score_dict = {}
# sentence_lengths_dict = {}
# term_weights_dict = {}
# sentence_positions_dict = {}
# numerical_score_ratios_dict = {}
# sentence_weights2_dict = {}
# thematic_word_feature_scores_dict = {}
# sentence_similarities_dict = {}
dictionary=[title_feature_score_dict ,
sentence_lengths_dict ,
term_weights_dict ,
sentence_positions_dict ,
numerical_score_ratios_dict ,
sentence_weights2_dict ,
thematic_word_feature_scores_dict ,
sentence_similarities_dict ]
# title_feature_scores = []
# sentence_lengths = []
# term_weights = []
# sentence_positions = []
# sentence_similarities = []
# numerical_score_ratios = []
# thematic_word_feature_scores = []
# sentence_weights2 = []
features = [
    "title_feature_scores",
    "sentence_lengths",
    "term_weights",
    "sentence_positions",
    "sentence_similarities",
    "numerical_score_ratios",
    "thematic_word_feature_scores",
    "sentence_weights2"
]


print('-----------------------------------------------------')

print('-----------------------------------------------------')

print('-----------------------------------------------------')
  


# sentence_1={}
# for i,feature in enumerate(feature):
#  main=dictionary[i]
#  sentence_1[feature]=main['sentence 1']
     

print('-----------------------------------------------------')

print('-----------------------------------------------------')

print('-----------------------------------------------------')
  
# print(sentence_1)  

def calculating_fuzzy_for_sentence(index_value,features,dictionary):
 
 name={}
 for i,feature in enumerate(features):
      
     main=dictionary[i]
     name[feature]=main[index_value]
 return name


fuzzy_values={}
for index,sentence in enumerate(sentences):

 fuzzy_values[sentence]=calculating_fuzzy_for_sentence(f'sentence {index+1}',features,dictionary)
 #fuzzy_for_each_sentence=fuzzy_values[f'sentence{index+1}']

 #print(f"for sentence{index+1} :   {fuzzy_for_each_sentence}")


 print('-----------------------------------------------------')

 print('-----------------------------------------------------')

 print('-----------------------------------------------------')


print('-----------------------------------------------------')

print('-----------------------------------------------------')

print('-----------------------------------------------------')

print('-----------------------------------------------------')

print('-----------------------------------------------------')

print('-----------------------------------------------------')


#print(fuzzy_values)
print('-----------------------------------------------------')

print('-----------------------------------------------------')

print('-----------------------------------------------------')

print('-----------------------------------------------------')

print('-----------------------------------------------------')

print('-----------------------------------------------------')











def generate_fuzzy_rules(features, linguistic_variables):
    fuzzy_rules = []

    # Generate combinations of linguistic variables for each feature
    variable_combinations = list(product(*linguistic_variables.values()))

    # Generate fuzzy rules using combinations of linguistic variables
    for combination in variable_combinations:
        antecedents = []
        consequence = ''

        # Form antecedents and consequence
        for i, feature in enumerate(features):
            antecedent = f"{feature} is {combination[i]}"
            antecedents.append(antecedent)

        # Add the generated fuzzy rule to the list
        fuzzy_rule = {
            "antecedents": antecedents,
            "consequence": consequence
        }
        fuzzy_rules.append(fuzzy_rule)

    return fuzzy_rules


def calculate_consequence(fuzzy_rules,converted_fuzzy_values  ):
    consequence_values = {}

    # Evaluate the fuzzy rules and determine the consequence
    for rule in fuzzy_rules:
        antecedent_results = []
        for antecedent in rule['antecedents']:
            feature, linguistic_value = antecedent.split(' is ')
            # print(feature, linguistic_value)
            # print(converted_fuzzy_values[feature])
            antecedent_result = converted_fuzzy_values[feature][linguistic_value]
            antecedent_results.append(antecedent_result)

        if min(antecedent_results) > 0:
            consequence = rule['consequence']
            if consequence in consequence_values:
                consequence_values[consequence] = max(consequence_values[consequence], min(antecedent_results))
            else:
                consequence_values[consequence] = min(antecedent_results)

    if consequence_values:
        max_consequence_value = max(consequence_values.values())
        consequence = [consequence for consequence, value in consequence_values.items() if value == max_consequence_value][0]
        return max_consequence_value

    return 'Unknown'  # Default consequence if none of the rules match


def arrange_sentences(sentences, fuzzy_rules, fuzzy_values):
    sentence_consequences = []

    for i,sentence in enumerate(sentences):
        # Calculate the consequence for each sentence
        print(fuzzy_values[sentence])
        consequence = calculate_consequence(fuzzy_rules,fuzzy_values[sentence])
        sentence_consequences.append((sentence, consequence))

    # Sort the sentences based on their consequence values in descending order
    sorted_sentences = sorted(sentence_consequences, key=lambda x: x[1], reverse=True)

    return sorted_sentences


# Define the features and linguistic variables
features = [
    "title_feature_scores",
    "sentence_lengths",
    "term_weights",
    "sentence_positions",
    "sentence_similarities",
    "numerical_score_ratios",
    "thematic_word_feature_scores",
    "sentence_weights2"
]

linguistic_variables = {
    "title_feature_scores": ["Low", "Medium", "High"],
    "sentence_lengths": ["Short", "Medium", "Long"],
    "term_weights": ["Low", "Medium", "High"],
    "sentence_positions": ["Start", "Middle", "End"],
    "sentence_similarities": ["Low", "Medium", "High"],
    "numerical_score_ratios": ["Low", "Medium", "High"],
    "thematic_word_feature_scores": ["Low", "Medium", "High"],
    "sentence_weights2": ["Low", "Medium", "High"]
}


fuzzy_rules = generate_fuzzy_rules(features, linguistic_variables)

 
#sentences = preprocess_text(text)


sorted_sentences = arrange_sentences(sentences, fuzzy_rules, fuzzy_values)


for sentence, consequence in sorted_sentences:
    print(f"Sentence: {sentence}")
    print(f"Consequence: {consequence}")
    print()


def fuzzy_text_summarization(sentences, N):
    
    
   
    # Step 3: Select the top N sentences
    top_sentences = sorted_sentences[:N]
    
    # Step 4: Preserve the original order of the sentences
    original_order_sentences = []
    for sentence, score in top_sentences:
        for i, original_sentence in enumerate(sentences):
            if sentence == original_sentence:
                original_order_sentences.append((sentence, score, i))
                break
    
    # Step 5: Sort the selected sentences based on their original positions
    sorted_original_order_sentences = sorted(original_order_sentences, key=lambda x: x[2])
    
    # Step 6: Get the final selected sentences in the original order
    final_sentences = [sentence for sentence, score, index in sorted_original_order_sentences]
    paragraph = ' '.join(final_sentences)
    
    return paragraph
    


summary = fuzzy_text_summarization(sentences, 4)
print(summary)