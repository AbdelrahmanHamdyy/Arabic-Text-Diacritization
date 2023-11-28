import re

# Convert using chr(diacritics[0])
# Ascii of The Diacritics
# Tanween bl fat7 - Tanween bl dam - Tanween bl kasr - Fat7a - Damma - Kasra - Shadda - Skoon
DIACRITICS = [1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618]
# Fasla
CONNECTOR = 1548
# Fasla Man2ota = 1563
OTHER = 'O'
# VOWEL_SYMBOLS = {'ٌ', 'ً', 'ٍ', 'ُ', 'َ', 'ِ', 'ْ', 'ٌّ', 'ّ'}
VOWEL_SYMBOLS = {chr(1614), chr(1615), chr(1616), chr(
    1618), chr(1617), chr(1611), chr(1612), chr(1613)}
VOWEL_REGEX = re.compile('|'.join(VOWEL_SYMBOLS))


# Bta5od kelma w tefsel el tshkeel 3n kol 7arf
def word_iterator(word):
    """
    This function takes a word (discretized or not) as an input and returns 
    a list of tuple where the first item is the character and the second
    item is the vowel_symbol. For example:
    >>> word_iterator('الْأَلْبَاب')
    [ ('ا', 'O'),
      ('ل', 'ْ'),
      ('أ', 'َ'),
      ('ل', 'ْ'), 
      ('ب', 'َ'), 
      ('ا', 'O'), 
      ('ب', 'O') ]
    As we can see, the symbol O stands for OTHER and it means that the character
    doesn't have an associated vowel symbol

    So Here the chars we have is something like letter-diacritic letter-diacritic 
    and some letters don't have diacritic 

    This is what is done here as, if the last char is a normal letter (not a diacritic) then we check for the next one
    if it's a diacritic then we will append the char with its diacritic, if not then we will append the char with Other(which stands for there is no diacritic)
    """
    output = []
    prev_char = word[0]
    #idx will start at 0
    #char we got here is the middle char as we have its prev element and its next element
    for idx, char in enumerate(word[1:]):
        print(word[idx])
        print(idx)
        print(char)
        print("--------")
        try:
            # first 1 because we skipped the first character
            # second 1 because it's the next character
            next_char = word[idx+1+1]
        except IndexError:  # will happen with the last character only
            next_char = ''
        #if the char is a diacritic then the prev char is mtshkl
        if char in VOWEL_SYMBOLS:
            #Here if we have exceeded the limit
            #Then the prev char is a letter
            #the current char is a diacritic
            #Then the end of the line
            #so add it
            if next_char == '' and prev_char not in VOWEL_SYMBOLS:
                output.append((prev_char, char))
            #Here the prev char isn't a vowel then it's a letter
            #And the Next char isn't a vowel
            #then we will append the prev char with the current char
            elif prev_char not in VOWEL_SYMBOLS and next_char not in VOWEL_SYMBOLS:
                output.append((prev_char, char))
            #To Handle if there is Shadda + Fat7a for example
            #Then prev char isn't a vowel
            #but the char and the next char are vowels
            #Then we will append the letter with Shadda+Fat7a
            elif prev_char not in VOWEL_SYMBOLS and next_char in VOWEL_SYMBOLS:
                output.append((prev_char, char+next_char))
        else:
            #Here we Found a letter Already
            #Then the prev char doesn't have any diacritic ahead
            #Then add this char with no diacritic
            if prev_char not in VOWEL_SYMBOLS:
                output.append((prev_char, OTHER))
            #We are in the end of the line
            if next_char == '':
                output.append((char, OTHER))
        prev_char = char
    return output


# Bt7sb accuracy (Btedeeha el kelma el sa7 wl predicted)
def evaluate_word(gold_word, predicted_word, analysis=False):
    """
    This function evaluate two input words:
    -> gold_word: represents the true discrentization of the word
    -> predicted_word: represents the model's discrentization of the word
    Then, this function should return the accuracy which depends on the following 
    formula which is:
                 number of correct tags
     accuracy = ------------------------
                 total number of tags
    """
    correct = 0.  # number of correct tags
    total_num = 0.  # total count of tags
    #Get the optimal tags
    gold_tags = [tag for _, tag in word_iterator(gold_word)]
    #Get the predicted tags
    predicted_tags = [tag for _, tag in word_iterator(predicted_word)]
    #Making sure that they both have the same length
    assert len(gold_tags) == len(predicted_tags), "Length isn't equal"
    #Looping over gold tags and predicted tags as we know they have the same length
    for gold_tag, predicted_tag in zip(gold_tags, predicted_tags):
        total_num += 1
        #Then if they are equal then increment the correct number by
        if gold_tag == predicted_tag:
            # print(gold_tag, predicted_tag)
            correct += 1.
    if analysis:
        return correct, total_num
    else:
        return correct/total_num


# Btsheel el tshkeel 5ales
def clean_word(word):
    """
    This function takes a word (discrentized or not) as an input and returns 
    the word itself without any discrentization.
    For example:
    >>> x = clean_word('الْأَلْبَاب')
    >>> x
    'الألباب'
    >>> type(x)
    'str'
    """
    return re.sub(VOWEL_REGEX, '', word)


def is_tashkeel(text):
    return any(ord(ch) in DIACRITICS for ch in text)


def clear_tashkeel(text):
    text = "".join(c for c in text if ord(c) not in DIACRITICS)
    return text


def get_diacritics():
    return "".join(chr(item)+"|" for item in DIACRITICS)[:-1]


def remove_undesired_characters(sentence):
    # Except: . ? ! ,
    pattern = re.compile('[\[\]\\/@#\$&%\^\+<=>(){}\*\|\`:;\'"\~_!\.\?\,؛-]')
    return re.sub(pattern, '', sentence)


def remove_html_tags(sentence):
    pattern = re.compile('<.*?>')
    return re.sub(pattern, '', sentence)


def remove_numbers(sentence):
    pattern = re.compile('[٠-٩0-9]')
    return re.sub(pattern, '', sentence)


def remove_english_letters(sentence):
    pattern = re.compile('[a-zA-Z]')
    return re.sub(pattern, '', sentence)


def data_cleaning(sentence):
    sentence = remove_english_letters(sentence)
    sentence = remove_numbers(sentence)
    sentence = remove_html_tags(sentence)
    sentence = remove_undesired_characters(sentence)
    return sentence


def tokenization(sentence):
    # tokenizer = Tokenizer(num_words=None,
    # filters='#$%&()*+-<=>@[\\]^_`{|}~\t\n',
    # lower = False, split = ' ')
    return sentence.split()


def preprocess(sentence):
    cleanedSentence = data_cleaning(sentence)
    finalSentence = tokenization(cleanedSentence)
    return finalSentence


if __name__ == '__main__':
    sentence = preprocess("الشَّ12هَادَةِ عَلَيْ[هِ مِثْلُY#!")
    print(sentence)
    for word in sentence:
        print(word_iterator(word))
        print(clean_word(word))
