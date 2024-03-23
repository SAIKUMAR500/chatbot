import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('popular', quiet=True)

# uncomment the following only the first time
# nltk.download('punkt') # first-time use only
# nltk.download('wordnet') # first-time use only

import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Reading in the corpus
with open('txt/chatbot.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

# Tokenization
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

# Preprocessing
lemmer = WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey", "morning", "evening")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me",
                      "good morning sir, have a good day ", 'good evening sir, i hope you had wonderful day',
                      'My name is jarvis ur personnel assistant ']


def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generating response
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response


# Speak the introductory message
intro_message = "Hi sir. My name is Jarvis. I will answer your queries about Chatbots. If you want to exit, type Bye!"
print("ROBO:", intro_message)
engine.say(intro_message)
engine.runAndWait()

flag = True

while (flag == True):
    user_response = input("You: ")
    user_response = user_response.lower()
    if (user_response != 'bye'):
        if (user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            response_message = "You are welcome.."
            print("Jaarvis:", response_message)
            engine.say(response_message)
            engine.runAndWait()
        else:
            if (greeting(user_response) != None):
                response_message = greeting(user_response)
                print("Jarvis:", response_message)
                engine.say(response_message)
                engine.runAndWait()
            else:
                robo_response = response(user_response)
                print("Jarvis:", robo_response)
                engine.say(robo_response)
                engine.runAndWait()
                sent_tokens.remove(user_response)
    else:
        flag = False
        response_message = "Bye! take care.."
        print("Jarvis:", response_message)
        engine.say(response_message)
        engine.runAndWait()
