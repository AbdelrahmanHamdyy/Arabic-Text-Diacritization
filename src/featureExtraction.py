from preprocessing import *
from utils import *
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import csv
from collections import OrderedDict
from transformers import BertModel, BertTokenizer

PATH = "../dataset/train.txt"
corpus = readFile(PATH)


# 1. Bag of words
# 2. TF-IDF
# 3. Word embeddings
# 4. Trainable embeddings
# 5. Contextual embeddings


# Word Embeddings
def extractWordEmbeddings():
    data = []
    for sentence in corpus:
        sentence = word_level_preprocess(sentence)
        data.append(sentence)
    model = Word2Vec(data, vector_size=100, min_count=1)
    print(model.wv['الْبَلَدِ'])
    print(model.wv.most_similar('الْبَلَدِ'))
    model.save('models/w2vmodel')


# TF-IDF
def TF_IDF():
    data = []
    for sentence in corpus:
        sentence = data_cleaning(sentence)
        data.append(sentence)
    tr_idf_model = TfidfVectorizer()
    tf_idf_vector = tr_idf_model.fit_transform(data)
    tf_idf_array = tf_idf_vector.toarray()
    words_set = tr_idf_model.get_feature_names_out()
    df_tf_idf = pd.DataFrame(tf_idf_array, columns=words_set)
    print(df_tf_idf)


# Contextual Embeddings
def bert_text_preparation(text, tokenizer):
    """
    Preprocesses text input in a way that BERT can interpret.
    """
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)
    # convert inputs to tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor = torch.tensor([segments_ids])
    return tokenized_text, tokens_tensor, segments_tensor


def get_bert_embeddings(tokens_tensor, segments_tensor, model):
    """
    Obtains BERT embeddings for tokens.
    """
    # gradient calculation id disabled
    with torch.no_grad():
        # obtain hidden states
        outputs = model(tokens_tensor, segments_tensor)
        hidden_states = outputs[2]
    # concatenate the tensors for all layers
    # use "stack" to create new dimension in tensor
    token_embeddings = torch.stack(hidden_states, dim=0)
    # remove dimension 1, the "batches"
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # swap dimensions 0 and 1 so we can loop over tokens
    token_embeddings = token_embeddings.permute(1, 0, 2)
    # intialized list to store embeddings
    token_vecs_sum = []
    # "token_embeddings" is a [Y x 12 x 768] tensor
    # where Y is the number of tokens in the sentence
    # loop over tokens in sentence
    for token in token_embeddings:
        # "token" is a [12 x 768] tensor
        # sum the vectors from the last four layers
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum.append(sum_vec)
    return token_vecs_sum


def extractContextualEmbeddings():
    model_name = "bert-base-multilingual-cased"
    model = BertModel.from_pretrained(model_name, output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Prepare Corpus
    # sentences = []
    # for sentence in corpus:
    #     sentence = data_cleaning(sentence)
    #     sentences.append(sentence)

    sentences = ["bank",
                 "he eventually sold the shares back to the bank at a premium.",
                 "the bank strongly resisted cutting interest rates.",
                 "the bank will supply and buy back foreign currency.",
                 "the bank is pressing us for repayment of the loan.",
                 "the bank left its lending rates unchanged.",
                 "the river flowed over the bank.",
                 "tall, luxuriant plants grew along the river bank.",
                 "his soldiers were arrayed along the river bank.",
                 "wild flowers adorned the river bank.",
                 "two fox cubs romped playfully on the river bank.",
                 "the jewels were kept in a bank vault.",
                 "you can stow your jewellery away in the bank.",
                 "most of the money was in storage in bank vaults.",
                 "the diamonds are shut away in a bank vault somewhere.",
                 "thieves broke into the bank vault.",
                 "can I bank on your support?",
                 "you can bank on him to hand you a reasonable bill for your services.",
                 "don't bank on your friends to help you out of trouble.",
                 "you can bank on me when you need money.",
                 "i bank on your help."
                 ]

    context_embeddings = []
    context_tokens = []
    for sentence in sentences:
        tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(
            sentence, tokenizer)
        list_token_embeddings = get_bert_embeddings(
            tokens_tensor, segments_tensors, model)
        # make ordered dictionary to keep track of the position of each   word
        tokens = OrderedDict()
        # loop over tokens in sensitive sentence
        for token in tokenized_text[1:-1]:
            # keep track of position of word and whether it occurs multiple times
            if token in tokens:
                tokens[token] += 1
            else:
                tokens[token] = 1
            # compute the position of the current token
            token_indices = [i for i, t in enumerate(
                tokenized_text) if t == token]
            current_index = token_indices[tokens[token]-1]
            # get the corresponding embedding
            token_vec = list_token_embeddings[current_index]
            # save values
            context_tokens.append(token)
            context_embeddings.append(token_vec)

    visualizeEmbeddings(context_tokens, context_embeddings)


def visualizeEmbeddings(context_tokens, context_embeddings):
    filepath = "models/embeddings.tsv"
    with open(filepath, 'w+') as file_metadata:
        for i, token in enumerate(context_tokens):
            file_metadata.write(token + '\n')
    with open(filepath, 'w+') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        for embedding in context_embeddings:
            writer.writerow(embedding.numpy())


if __name__ == '__main__':
    extractWordEmbeddings()
    TF_IDF()
    extractContextualEmbeddings()
