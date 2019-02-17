import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize, sent_tokenize
from nltk.tokenize.moses import MosesDetokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
import matplotlib
import csv
import string
import re
import sys

# Function that Loads and Cleans Data
def initialize():
    numOfRows = 140
    tempFlag = False
    # df = pd.read_csv("warrantyFailModes.csv", encoding="ISO-8859-1")# Load file
    df = pd.read_csv("MockWarrantyClaims.csv", encoding="ISO-8859-1", nrows=numOfRows) # Load file
    #df = df.drop(['PIN_NUM_13', 'PIN_NUM_17', 'BUS_PROD_SERIES', 'WRNTY_CLM_NUM', 'REGION_CD', 'FAILURE_DT', 'CREDIT_DT', 'MFG_DT', 'MACHINE_HR_CNT', 'CORRECT_PART_NUM', 'PART_NAME_DSC', 'PART_DSC'], axis = 1) # axis = 1 is to remove columns, otherwise currentClaims. , 'CAUSE_TXT (reason)', 'CORRECT_TXT (action)'

    # Delete all rows with the following failure mode 
    df = df[df['Failure Mode Original'].str.contains("Internal Gearcase Failure - Primary failed part unknown") == False]
    return df

# Function that Tokenizes Each Piece of Text
def applyWordTokenization(string):
    tokenizedWord = word_tokenize(string)
    return tokenizedWord

def applySentenceTokenization(string):
    tokenizedSentence = sent_tokenize(string)
    return tokenizedSentence

def wordTokenToString(tokenizedWord):
    detokenizer = MosesDetokenizer()
    return detokenizer.detokenize(tokenizedWord, return_str=True)

# Function that applies Parts of Speech Tagging
def applyPOS_Tag(tokenizedWord):
    '''
    POS tag list:

    CC	coordinating conjunction
    CD	cardinal digit
    DT	determiner
    EX	existential there (like: "there is" ... think of it like "there exists")
    FW	foreign word
    IN	preposition/subordinating conjunction
    JJ	adjective	'big'
    JJR	adjective, comparative	'bigger'
    JJS	adjective, superlative	'biggest'
    LS	list marker	1)
    MD	modal	could, will
    NN	noun, singular 'desk'
    NNS	noun plural	'desks'
    NNP	proper noun, singular	'Harrison'
    NNPS	proper noun, plural	'Americans'
    PDT	predeterminer	'all the kids'
    POS	possessive ending	parent's
    PRP	personal pronoun	I, he, she
    PRP$	possessive pronoun	my, his, hers
    RB	adverb	very, silently,
    RBR	adverb, comparative	better
    RBS	adverb, superlative	best
    RP	particle	give up
    TO	to	go 'to' the store.
    UH	interjection	errrrrrrrm
    VB	verb, base form	take
    VBD	verb, past tense	took
    VBG	verb, gerund/present participle	taking
    VBN	verb, past participle	taken
    VBP	verb, sing. present, non-3d	take
    VBZ	verb, 3rd person sing. present	takes
    WDT	wh-determiner	which
    WP	wh-pronoun	who, what
    WP$	possessive wh-pronoun	whose
    WRB	wh-abverb	where, when
    '''
    posTag = nltk.pos_tag(tokenizedWord) # Tag each token
    return posTag

# Function that applies Chunking
def applyChunking(tokenizedWord):
    '''
    Depending on your goals, you may use the binary option how you see fit. Here are the types of Named Entities that you can get if you have binary as false:

    NE Type and Examples
    ORGANIZATION - Georgia-Pacific Corp., WHO
    PERSON - Eddy Bonte, President Obama
    LOCATION - Murray River, Mount Everest
    DATE - June, 2008-06-29
    TIME - two fifty a m, 1:30 p.m.
    MONEY - 175 million Canadian Dollars, GBP 10.40
    PERCENT - twenty pct, 18.75 %
    FACILITY - Washington Monument, Stonehenge
    GPE - South East Asia, Midlothian
    '''
    chunkedWord = ne_chunk(tokenizedWord)
    #chunkedWord = ne_chunk(tokenizedWord).draw()
    return chunkedWord

def get_wordnet_pos(pos_tag):
    if pos_tag[1].startswith('J'):
        return (pos_tag[0], wordnet.ADJ)
    elif pos_tag[1].startswith('V'):
        return (pos_tag[0], wordnet.VERB)
    elif pos_tag[1].startswith('N'):
        return (pos_tag[0], wordnet.NOUN)
    elif pos_tag[1].startswith('R'):
        return (pos_tag[0], wordnet.ADV)
    else:
        return (pos_tag[0], wordnet.NOUN)

'''-------------------------------------------------------------------'''
''' POTENTIAL DELETIONS '''
'''
# Function that Lemmaizes words
def applyLemmatizer(tokenizedWord):
    newtokenizedWord = []
    lemmatizer = WordNetLemmatizer()
    for word in tokenizedWord:
        newtokenizedWord.append(lemmatizer.lemmatize(word))
    return newtokenizedWord
'''

'''
# Function that Stems Words
def applyStemming(tokenizedWord):
    newtokenizedWord = []
    ps = PorterStemmer()
    for word in tokenizedWord:
        newtokenizedWord.append(ps.stem(word))
    return newtokenizedWord
'''

'''
# Function that Removes Custom Stop Words
def applyCustomStopWords(tokenizedWord):
    # Add custom stop words here:
    customStopWords = ['the'] # ['removed', 'snap', 'ring', 'drained', 'hydraulic']
    newtokenizedWord = []
    for word in tokenizedWord:
        if word not in customStopWords:
            newtokenizedWord.append(word)
    return newtokenizedWord
'''

'''
# Returns tokenized lower case word
def applyLowerCase(tokenizedWord):
    # new tokenized word
    newtokenizedWord = []
    for word in tokenizedWord:
        newtokenizedWord.append(word.lower())
    #words = [word.lower() for word in tokenizedWord]
    return newtokenizedWord
'''

'''
# Function that Removes Stop Words
def applyStopWords(tokenizedWord):
    # List of Stop Words:
    # {'most', 'more', 'shan', 'won', "you've", 'in', 'their', 'all', 'once', 'can', 'o', 'not', "shan't", 'did', "didn't", 'this', 'from', 'as', 'above', 'any', 't', 'than', 'mustn', 'against', 'only', "don't", 'then', "couldn't", 'myself', 'you', 'during', 'weren', "needn't", 've', 'is', 'hadn', "mightn't", 'she', 'before', 'y', 'wasn', "weren't", 'those', 'both', 'very', 'be', 'yourself', 'your', 'a', 'no', "that'll", 're', 'ain', 'i', 'herself', 'itself', 'here', 'just', 'haven', 'on', 'hers', 'our', 'didn', "you'd", 'themselves', 'because', 'these', 'when', 'few', "won't", 'was', 'or', 'he', 'for', 'other', 'have', 'until', 'had', 'to', 'off', "it's", 'out', 'wouldn', "wasn't", 'them', 'they', "haven't", 'so', 'does', 'were', "you're", 'has', 'there', 'd', "hasn't", 'down', 'why', 'theirs', "mustn't", "should've", "wouldn't", 'below', 'ma', 'hasn', 'himself', 'each', 'of', 'will', 'own', 'over', 'into', 'yours', 'same', "she's", 'such', 'too', 'yourselves', "you'll", 'which', 'the', 'where', 'we', 'him', 'couldn', 'with', 'again', 'ours', 'been', 'after', 'needn', 'do', 'some', 'isn', 'doing', 'having', 'about', 'me', 'under', 'shouldn', 'an', 'should', 'am', 'aren', 'm', 'and', 'how', "aren't", "doesn't", 'through', "isn't", 'between', 'now', 's', 'it', 'doesn', 'by', 'who', 'her', 'up', 'its', 'my', 'll', 'that', "shouldn't", "hadn't", 'if', 'mightn', 'but', 'nor', 'what', 'whom', 'are', 'further', 'ourselves', 'being', 'his', 'at', 'while', 'don'}
    #
    # print(string.punctuation) # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

    stopWords = set(stopwords.words('english')) # Save stop words
    newtokenizedWord = []
    for word in tokenizedWord:
        if word not in stopWords:
            newtokenizedWord.append(word)
    return newtokenizedWord
'''

# Call Functions in Here
def main():
    myThreshold = 0.7

    # Data Frame/CSV Routine
    dataFrame = initialize() # To load cleaned Data
    dataFrame['Concactenate'] = dataFrame.CMPLNT_TXT + ' ' + dataFrame.CAUSE_TXT + ' ' + dataFrame.CORRECT_TXT
    dataFrame['Filtered Concactenation'] = 0 # Create New Column
    dataFrame['Ratio'] = 0
    dataFrame['Failure Mode'] = 0
    dataFrame['Failure Mode Classified'] = 0
    dataFrame['Classification Accuracy'] = 0

    docString = "" # This will contain cumulative string for frequency count of words in entire document

    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(string.punctuation) #  adding punctuations list of stop words
    stopwords.append('') # simply adding space/whitespace to list of stop words
    customStopWords = ['the'] # ['removed', 'snap', 'ring', 'drained', 'hydraulic']
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    # failureModeList = ['Scavenger Pump Seal Leak', 'Transmission Output Shaft Seal Leak', 'Lower Shaft Seal Leak', 'Upper Shaft Seal Leak', 'Drive Shaft Oil Leak', 'Tractor PTO Oil Leak', 'Hydraulic Pump Drive Leak']
    failureModeList = ['Broken Pinion Bolt', 'Internal Gearcase Failure Primary failed part unknown', 'Cracked Housing', 'Loose Bolt', 'Differential Bearing Vibration', 'Cracked Hub', 'Kingpin Bearing Failure', 'MFWD Input Nut Loose', 'Kingpin Loose wheel hubs', 'Kingpin Failure', 'MFWD Input Bearing Loss of Power', 'MFWD Input Bearing Vibration', 'MFWD Input Bearing Failure', 'Wheel Hub Bearing Leak', 'Lock Nut Unscrewed from shaft', 'U-joint Bearings/Cross', 'Hub Threads', 'End Clamps', 'Wheel Hub Bearing Loose', 'Wheel Hub Bearing Vibration' 'Final Drive Input Bearing or Seal Leak', 'Wheel Hub Bearing Failure'] # 142

    for currentFailureMode in range(len(failureModeList)):
        # Initialize Arrays that will temporarily store string that will eventually be copied over to CSV
        myOutput = []
        myFilteredToken = []
        myRatio = []
        myFailureMode = []
        myFailModeMatchFlag = []
        failModeMatchCnt = 0
        # Go through each Claim One-by-One
        for currentClaim in range(len(dataFrame)):
            #print(currentClaim + 2) # debug excel row
            #print(dataFrame.iat[currentClaim,8]) # debug print(dataFrame.iloc[currentClaim]['Concactenate'])

            # Check if a and b are matches...
            #print(failureModeList[currentFailureMode])

            a = failureModeList[currentFailureMode] # "Upper Transmission Oil Leak"
            if (str(dataFrame.iloc[currentClaim]['Concactenate']) == 'nan'): # If word is nan or na, it causes issues
                b = "NOT APPLICABLE"
            else:
                b = dataFrame.iloc[currentClaim]['Concactenate'] # Otherwise, save current claim into b

            pos_a = map(get_wordnet_pos, applyPOS_Tag(applyWordTokenization(a))) # apply function to each element in array. Output is a map object
            pos_b = map(get_wordnet_pos, applyPOS_Tag(applyWordTokenization(b)))
            # print(list(pos_b)) # print: [('oil', 'n'), ('leak', 'n'),

            lemmae_a = []
            lemmae_b = []
            # Go through each token in parts of speech array
            for token, pos in pos_a:
                if ((pos == wordnet.NOUN or pos == wordnet.VERB or pos == wordnet.ADJ) and token.lower().strip(string.punctuation)) not in (stopwords and customStopWords):
                    lemmae_a.append(lemmatizer.lemmatize(token.lower().strip(string.punctuation), pos))
            for token, pos in pos_b:
                if ((pos == wordnet.NOUN or pos == wordnet.VERB or pos == wordnet.ADJ) and token.lower().strip(string.punctuation)) not in (stopwords and customStopWords):
                    lemmae_b.append(lemmatizer.lemmatize(token.lower().strip(string.punctuation), pos))
            #print(lemmae_a)
            #print(lemmae_b)

            # Calculate Jaccard similarity
            # The "set" function makes sure that duplicates words are considered only once
            ratio = len(set(lemmae_a).intersection(lemmae_b)) / float(len(set(lemmae_a))) # ratio = len(set(lemmae_a).intersection(lemmae_b)) / float(len(set(lemmae_a).union(lemmae_b)))
            # print(ratio)


            # Save data to python arrays for eventual transfer to dataFrame
            filteredString = wordTokenToString(lemmae_b) # Convert tokenized text to string
            myOutput.append(filteredString) # save string to output array
            myFilteredToken.append(lemmae_b) # save filtered tokenized words to myFilteredToken array
            #myRatio.append(ratio) # add percent match of words to myRatio array
            print(ratio)

            if ratio > myThreshold and dataFrame.iloc[currentClaim]['Ratio'] < myThreshold: # If current claim has high percentage match to current failure mode and current claim has a lower percentage match to the last failure mode
                myFailureMode.append(a) # Bucket claim into current failure mode
                myRatio.append(ratio) # add percent match of words to myRatio array
            elif ratio > myThreshold and dataFrame.iloc[currentClaim]['Ratio'] > myThreshold: # if current claim has high percentage match to current failure mode and current claim has high precentage match with last failure mode as well...
                if ratio > dataFrame.iloc[currentClaim]['Ratio']: # ... and current claim matches better with the current failure mode than the previous failure mode
                    myFailureMode.append(a) # Buck claim into current failure mode
                    myRatio.append(ratio) # add percent match of words to myRatio array
                elif ratio < dataFrame.iloc[currentClaim]['Ratio']: # If current failure mode has lower percentage match to current failure mode than the last failure mode
                    myFailureMode.append(dataFrame.iloc[currentClaim]['Failure Mode'])
                    myRatio.append(dataFrame.iloc[currentClaim]['Ratio']) # add percent match of words to myRatio array
                    # Bucket claim into last failure mode
                else: # If current match and previous match to failure mode are matching in ratio
                    # Use current failure mode
                    myFailureMode.append(a) # probably need to create it's own bucket
                    myRatio.append(ratio)
            else:
                #myFailureMode.append('Unknown')
                #myRatio.append(float(0)) # add percent match of words to myRatio array
                myFailureMode.append(dataFrame.iloc[currentClaim]['Failure Mode'])
                myRatio.append(dataFrame.iloc[currentClaim]['Ratio']) # add percent match of words to myRatio array

            if myFailureMode[currentClaim] == dataFrame.iloc[currentClaim]['Failure Mode Original']:
                myFailModeMatchFlag.append(True)
                failModeMatchCnt = failModeMatchCnt + 1
            else:
                myFailModeMatchFlag.append(False)


            docString = docString + " " + filteredString # cumulative string of text throughout document

        # copy array to .csv
        dataFrame['Filtered Concactenation'] = myOutput
        dataFrame['Ratio'] = myRatio
        dataFrame['Failure Mode'] = myFailureMode
        dataFrame['Failure Mode Classified'] = myFailModeMatchFlag
        dataFrame['Classification Accuracy'] = float(failModeMatchCnt/len(myFailModeMatchFlag))

        for col in dataFrame.columns:
            if 'Unnamed' in col:
                del dataFrame[col]
        dataFrame.to_csv('cleanedWarrantyFailModes.csv') # Writing to CSV

        print(failModeMatchCnt)
        print(len(myFailModeMatchFlag))



        #print(dataFrame.iloc[0]['Ratio'])
        # Word Count Charts
        wordCount = nltk.FreqDist(myFilteredToken[3]) # Word Count of one element
        #wordCount.plot(50, cumulative=False)
            # print(wordCount.most_common(15)) #print(wordCount["coupler"])
        docStringToken = applyWordTokenization(docString)
        totalWordCount = nltk.FreqDist(docStringToken)
        #totalWordCount.plot(50, cumulative=False)


# Invoke Main Function
main()
