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
    # idx will start at 0
    # char we got here is the middle char as we have its prev element and its next element
    for idx, char in enumerate(word[1:]):
        try:
            # first 1 because we skipped the first character
            # second 1 because it's the next character
            next_char = word[idx+1+1]
        except IndexError:  # will happen with the last character only
            next_char = ''
        # if the char is a diacritic then the prev char is mtshkl
        if char in VOWEL_SYMBOLS:
            # Here if we have exceeded the limit
            # Then the prev char is a letter
            # the current char is a diacritic
            # Then the end of the line
            # so add it
            if next_char == '' and prev_char not in VOWEL_SYMBOLS:
                output.append((prev_char, char))
            # Here the prev char isn't a vowel then it's a letter
            # And the Next char isn't a vowel
            # then we will append the prev char with the current char
            elif prev_char not in VOWEL_SYMBOLS and next_char not in VOWEL_SYMBOLS:
                output.append((prev_char, char))
            # To Handle if there is Shadda + Fat7a for example
            # Then prev char isn't a vowel
            # but the char and the next char are vowels
            # Then we will append the letter with Shadda+Fat7a
            elif prev_char not in VOWEL_SYMBOLS and next_char in VOWEL_SYMBOLS:
                output.append((prev_char, char+next_char))
        else:
            # Here we Found a letter Already
            # Then the prev char doesn't have any diacritic ahead
            # Then add this char with no diacritic
            if prev_char not in VOWEL_SYMBOLS:
                output.append((prev_char, OTHER))
            # We are in the end of the line
            if next_char == '':
                output.append((char, OTHER))
        prev_char = char
    return output


# This function is used to get aware of the surrounding letters of the current letter
# as we can get n letters before this char as they may change the values of diacritic this char can take
def include_ngram_letters(word, index, pair, n):
    # Firstly this letters array contains the previous chars in the word
    letters = []
    # But Firstly we need to remove the diacritics themselves from the word
    word = clean_word(word)
    # Then we should get the position where we should stop at
    # If the previous n chars are out of the limit of the word then we should stop at letter 0
    stoppingIndex = index-n+1
    # Checking that we should stop at letter 0
    if (stoppingIndex < 0):
        stoppingIndex = 0
    # Then looping from the index we are in backward until the stopping index position
    while (index >= stoppingIndex):
        letters.insert(0, word[index])
        index -= 1
    # Then We fill the gaps with * which means that no chars are there
    while (len(letters) < n):
        letters.insert(0, '*')
    return (letters, pair[1])


# This Function is used to get the ngram key itself
# as this key consists of ((previous n-1 values,char itself),Diacritic of the the char itself)
# So we will be aware of what is surrounding
def ngram_key_generator(word, n):
    # We get firstly the diacritic of each letter in the given array
    one_letter_diacritic_list = word_iterator(word)
    n_letters_diacritic_list = []
    # Then we loop over that array and for each index inside it we get the previous n chars
    # Then Generating the final array containing the (n Chars ,diacritic of the last char)
    for index, element in enumerate(one_letter_diacritic_list):
        n_letters_diacritic_list.append(
            include_ngram_letters(word, index, element, n))
    return n_letters_diacritic_list


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
    # Get the optimal tags
    gold_tags = [tag for _, tag in word_iterator(gold_word)]
    # Get the predicted tags
    predicted_tags = [tag for _, tag in word_iterator(predicted_word)]
    # Making sure that they both have the same length
    assert len(gold_tags) == len(predicted_tags), "Length isn't equal"
    # Looping over gold tags and predicted tags as we know they have the same length
    for gold_tag, predicted_tag in zip(gold_tags, predicted_tags):
        total_num += 1
        # Then if they are equal then increment the correct number by
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
    pattern = re.compile('[\[\]\\/@#\$&%\^\+<=>(){}\*\|\`:;\'"\~_!\.\?؟\,،؛-]')
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
    # Remove extra spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = ' '.join(sentence.split())
    return sentence


def tokenization(sentence):
    # tokenizer = Tokenizer(num_words=None,
    # filters='#$%&()*+-<=>@[\\]^_`{|}~\t\n',
    # lower = False, split = ' ')
    return sentence.split()


def word_level_preprocess(sentence):
    cleanedSentence = data_cleaning(sentence)
    finalSentence = tokenization(cleanedSentence)
    return finalSentence


def characters_with_diacritics_tuples(sentence):
    sentence = word_level_preprocess(sentence)
    result = []
    for word in sentence:
        result += word_iterator(word.strip())
    return result


def remove_diacritics(sentence):
    sentence = word_level_preprocess(sentence)
    result = ""
    for word in sentence:
        result += clean_word(word.strip()) + " "
    return result


def ngram(sentence, n):
    sentence = word_level_preprocess(sentence)
    result = []
    for word in sentence:
        result.append(ngram_key_generator(word.strip(), n))

    return result


def separate_words_and_diacritics(sentence):
    sentence = word_level_preprocess(sentence)
    letters = []
    diacritics = []
    for word in sentence:
        prev_char = word[0]
        for idx, char in enumerate(word[1:]):
            try:
                next_char = word[idx + 1 + 1]
            except IndexError:
                next_char = ''
            if char in VOWEL_SYMBOLS:
                if prev_char not in VOWEL_SYMBOLS:
                    letters.append(prev_char)
                    if next_char == '' or next_char not in VOWEL_SYMBOLS:
                        diacritics.append(char)
                    elif next_char in VOWEL_SYMBOLS:
                        diacritics.append(char + next_char)
            else:
                if prev_char not in VOWEL_SYMBOLS:
                    letters.append(prev_char)
                    diacritics.append(OTHER)
                if next_char == '':
                    letters.append(char)
                    diacritics.append(OTHER)
            prev_char = char

    return letters, diacritics


if __name__ == '__main__':
    sentence = "الشَّ12هَادَةِ عَلَيْ[هِ   مِثْلُY#!"
    print("Test Sentence:", sentence)
    print("----------------------------------------------")
    print("Cleaned Sentence:", data_cleaning(sentence))
    print("----------------------------------------------")
    print("Word Level:", word_level_preprocess(sentence))
    print("----------------------------------------------")
    letters, diacritics = separate_words_and_diacritics(sentence)
    print("Letters:", letters)
    print("Diacritics:", diacritics)
    print("----------------------------------------------")
    print("Diacritics Removed:", remove_diacritics(sentence))
    print("----------------------------------------------")
    print("Characters with Diacritics:",
          characters_with_diacritics_tuples(sentence))
    print("----------------------------------------------")
    print("NGram:", ngram(sentence, 3))

#############################################################################################################
# Some Features should take care of the position of the letter
# As some diacritics like double fat7a and double kasra and double damma should be at the end of the word


# Now We Have 15 Classes
# 1-> Tanween bl fat7a (Lazm fe akher letter)
# 2-> Tanween bl damma (Lazm fe akher letter)
# 3-> Tanween bl kasra (Lazm fe akher letter)
# 4-> Fat7a
# 5-> damma
# 6-> Kasra
# 7-> Shadda
# 8-> Skoon
# 9-> Shadda m3 Fat7a
# 10-> Shadda m3 damma
# 11-> Shadda m3 kasra
# 12-> Shadda m3 tanween bl fat7a (Lazm fe akher letter)
# 13-> Shadda m3 tanween bl damma (Lazm fe akher letter)
# 14-> Shadda m3 tanween bl kasra (Lazm fe akher letter)
# 15-> No Diacritic

# (\b[\w]+\b)
