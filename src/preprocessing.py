import re
from lang_trans.arabic import buckwalter
from constants import *

char_to_index = {char: i + 1 for i, char in enumerate(basic_arabic_letters)}
diacritic_to_index = {diacritic: i for i, diacritic in enumerate(DIACRITICS)}
index_to_diacritic = {index: diacritic for diacritic, index in diacritic_to_index.items()}

def combine_whitespaces(text):
    return re.sub(WHITESPACES_PATTERN, " ",text)

def readFile(path):
    sentences = []
    with open(path, 'r', encoding='utf-8') as file:
       for line in file:
           sentences.append(line.strip())

    return sentences

def get_valid_arabic_letters(text):
    text = list(filter(lambda char: char in basic_arabic_letters,text))
    return combine_whitespaces(''.join(text))

def get_valid_arabic_text(text):
    text = list(filter(lambda char: char in VALID_ARABIC_CHARS,text))
    return combine_whitespaces(''.join(text))

def get_valid_arabic_text_without_punctuation(text):
    text = list(filter(lambda char: char in VALID_ARABIC_CHARS_WITHOUT_PUNCTUATION,text))
    return combine_whitespaces(''.join(text))

def separate_words_to_char(sentence):
    sentence = get_valid_arabic_text_without_punctuation(sentence)

    letters = []
    for word in sentence.split():
        for char in word:
            letters.append(char)

    return letters

def get_sentences_window(sentence):
    startIdx = 0
    sentences = []
    sentence = get_valid_arabic_text_without_punctuation(sentence)
    while(startIdx < len(sentence) and startIdx != -1):

        finalIdx = startIdx + SENTENCE_WINDOW if startIdx + SENTENCE_WINDOW < len(sentence) else -1

        if finalIdx != -1 and finalIdx < len(sentence) and sentence[finalIdx] in MAIN_DIACRITICS:
            finalIdx -= 1

        while(finalIdx != -1 and sentence[finalIdx] != ' '):
            finalIdx-=1
        
        pre_modified_sentence = sentence[startIdx:finalIdx] if finalIdx != -1 else sentence[startIdx:len(sentence)]
        startIdx = finalIdx
        sentences.append(pre_modified_sentence)

    return sentences

def get_all_windows(sentence):
    windows =[]
    sentence = get_valid_arabic_text_without_punctuation(sentence)
    sentence = sentence.split()
    for i in range(len(sentence)):
        start_idx = max(0,i-WINDOW_SIZE_BEFORE)
        last_index = min(len(sentence),i+WINDOW_SIZE_AFTER+1)
        windows.append(sentence[start_idx:last_index])
    return windows

def get_splitted_sentences(sentence):
    sentence = get_valid_arabic_text(sentence)
    windows = get_sentences_window(sentence)
    return windows

def separate_words_and_diacritics(sentence):
    sentences = get_splitted_sentences(sentence)
    final_chars = []
    final_diacritics = []

    for sentence in sentences:
        output_chars = []
        output_diacritics = []
        for word in sentence.split():
            letters = []
            diacritics = []
            prev_char = word[0]
            if len(word) == 1:
                letters.append(prev_char)
                diacritics.append(OTHER)
            else:
                for idx, char in enumerate(word[1:]):
                    try:
                        next_char = word[idx + 1 + 1]
                    except IndexError:
                        next_char = ''
                    if char in DIACRITICS:
                        if prev_char not in DIACRITICS:
                            letters.append(prev_char)
                            if next_char == '' or next_char not in DIACRITICS:
                                diacritics.append(char)
                            elif next_char in DIACRITICS:
                                # print(char+next_char)
                                diacritics.append(char + next_char)
                    else:
                        if prev_char not in DIACRITICS:
                            letters.append(prev_char)
                            diacritics.append(OTHER)
                        if next_char == '':
                            letters.append(char)
                            diacritics.append(OTHER)
                    prev_char = char

            if len(letters):
                output_chars.append(letters)
                output_diacritics.append(diacritics)

        final_chars.append([char for word in output_chars for char in word])
        final_diacritics.append([diacritic for word in output_diacritics for diacritic in word])

    final_chars = [item for item in final_chars if len(item)]
    final_diacritics = [item for item in final_diacritics if len(item)]
    return final_chars, final_diacritics

def run_buckwalter(sentence):
    sentence = get_valid_arabic_text(sentence)
    output = []
    for word in sentence.split():
        output.append(buckwalter.transliterate(word))
    return output
    