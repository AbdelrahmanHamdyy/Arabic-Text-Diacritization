from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertModel, BertTokenizer
from collections import OrderedDict
import torch
import csv
import pandas as pd
from lang_trans.arabic import buckwalter
from preprocessing import *
from constants import *

corpus = readFile(TRAIN_PATH)

def get_buckwalter_data(space=False):
    data = []
    for sentence in corpus:
        sentence = run_buckwalter(sentence)
        char_list, _ = separate_words_and_diacritics(sentence)
        char_list = ["".join(sen) for sen in char_list]
        data.append(" ".join(char_list) if space else char_list)
    return data


def fasttext_word_embeddings():
    data = get_buckwalter_data()
    # Defining values for parameters
    embedding_size = 100
    window_size = 20
    min_word = 5
    down_sampling = 1e-2

    fast_Text_model = FastText(
                                vector_size=embedding_size,
                                window=window_size,
                                min_count=min_word,
                                sample=down_sampling,
                                workers=4,
                                epochs=50,
                                seed=42,
                                sg=1)
    fast_Text_model.build_vocab(data, progress_per=10000)
    fast_Text_model.train(data, total_examples=fast_Text_model.corpus_count, epochs=50, report_delay=1)
    
    fast_Text_model.save("./models/ft_model")
    print(fast_Text_model.wv[buckwalter.transliterate("ياكل")])
    print(fast_Text_model.wv.similarity(buckwalter.transliterate("احمد"), buckwalter.transliterate("محمد")))
    
def tf_idf():
    data = get_buckwalter_data(space=True)
    
    tr_idf_model = TfidfVectorizer(lowercase=False)
    tf_idf_vector = tr_idf_model.fit_transform(data)
    
    words_set = tr_idf_model.get_feature_names_out()
    
    df_tf_idf = pd.DataFrame(columns=words_set)
    chunk_size=1000
    for i in range(0, tf_idf_vector.shape[0], chunk_size):
            end_idx = min(i + chunk_size, tf_idf_vector.shape[0])
            tf_idf_chunk = tf_idf_vector[i:end_idx].toarray()
            chunk_df = pd.DataFrame(tf_idf_chunk, columns=words_set)
            df_tf_idf = pd.concat([df_tf_idf, chunk_df], ignore_index=True)
    df_tf_idf.to_csv('models/tf_idf.csv', index=False)
    
    # Read the DataFrame from the CSV file
    df_from_csv = pd.read_csv('your_dataframe.csv')
    # Accessing TF-IDF values for the word 'example'
    print(df_from_csv['example'])
    
def bag_of_words():
    data = get_buckwalter_data(space=True)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data)
    feature_names = vectorizer.get_feature_names_out()
    X_array = X.toarray()

    df = pd.DataFrame(data=X_array, columns=feature_names, index=data)
    df.to_csv('models/bag_of_words.csv', index=False)
    
def bert_text_preparation(text, tokenizer):
    """
    Preprocesses text input in a way that BERT can interpret.
    """
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.encode(marked_text, max_length=512, truncation=True, padding=True)
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

def visualize_embeddings(context_tokens, context_embeddings):
    filepath = "models/embeddings.tsv"
    with open(filepath, 'w+') as file_metadata:
        for i, token in enumerate(context_tokens):
            file_metadata.write(token + '\n')
    with open(filepath, 'w+') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        for embedding in context_embeddings:
            writer.writerow(embedding.numpy())
            
def contextual_embeddings():
    data = get_buckwalter_data(space=True)
    model_name = "bert-base-multilingual-cased"
    model = BertModel.from_pretrained(model_name, output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    context_embeddings = []
    context_tokens = []
    for sentence in data:
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

    visualize_embeddings(context_tokens, context_embeddings)