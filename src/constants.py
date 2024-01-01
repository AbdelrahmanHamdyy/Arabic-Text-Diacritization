import re
import pickle

# Window sizes
SENTENCE_WINDOW = 600
WINDOW_SIZE_BEFORE = 2
WINDOW_SIZE_AFTER = 3

# Diacritics and punctuations
OTHER = ' '
DIACRITICS = [OTHER, "َ", "ً", "ُ", "ٌ", "ِ", "ٍ", "ْ", "ّ", "َّ", "ًّ", "ُّ", "ٌّ", "ِّ", "ٍّ"]
PUNCTUATIONS = [".", "،", ":", "؛", "؟"]
MAIN_DIACRITICS = None
with open("./utils/diacritics.pickle","rb") as file:
    MAIN_DIACRITICS = list(pickle.load(file))

# Model parameters
EMBEDDING_DIM = 300
HIDDEN_SIZE = 512
NUM_LAYERS = 3
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
BATCH_SIZE = 64
PAD = 15

# Arabic Letters
basic_arabic_letters = None
with open("./utils/arabic_letters.pickle","rb") as file:
    basic_arabic_letters = list(pickle.load(file))
VALID_ARABIC_CHARS = basic_arabic_letters + MAIN_DIACRITICS  + PUNCTUATIONS + [' ']
VALID_ARABIC_CHARS_WITHOUT_PUNCTUATION = basic_arabic_letters + MAIN_DIACRITICS + [' ']
WHITESPACES_PATTERN = re.compile("\s+")
    
# Diacritic to Id
diacritic2id = None
with open("./utils/diacritic2id.pickle", 'rb') as file:
    diacritic2id = pickle.load(file)
    
# Lengths
VOCAB_SIZE = len(basic_arabic_letters) + 1
LABELS_SIZE = len(DIACRITICS)

# Paths
RNN_PATH = "../trained_models/rnn.pth"
CNN_PATH = "../trained_models/cnn.pth"
LSTM_PATH = "../trained_models/lstm.pth"
CRF_PATH = "../trained_models/crf.pth"

TRAIN_PATH = "../dataset/train.txt"
VAL_PATH = "../dataset/val.txt"

TEST_PATH = "../dataset/test.txt"
TEST2_PATH = "../dataset/test2.txt"
TEST3_PATH = "../dataset/test3.txt"
TEST4_PATH = "../dataset/test4.txt"

SAMPLE_TEST_PATH = "./test/sample_test_no_diacritics.txt"
SAMPLE_TEST_GOLD_PATH = "./test/sample_test_set_gold.csv"
OUTPUT_PATH = "./test/output.csv"
GOLD_PATH = "./test/test_set_gold.csv"
