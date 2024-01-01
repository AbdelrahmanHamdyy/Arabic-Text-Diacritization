import csv
import torch
import pandas as pd
from preprocessing import *
from train import RNN
from constants import *
    
def read_test_set(path):
    test_corpus = readFile(path)

    X_test = []
    Y_test = []

    for sentence in test_corpus:
        char_list, diacritics_list = separate_words_and_diacritics(sentence.strip())
        for i in range(len(char_list)):
            X_test.append(char_list[i])
            Y_test.append(diacritics_list[i])
            
    return X_test, Y_test

def write_gold_output(Y_test):
    data = []
    i = 0

    for sentence in Y_test:
        for diacritic in sentence:
            data.append({"ID": i, "label": diacritic2id["" if diacritic == " " else diacritic]})
            i += 1

    with open(GOLD_PATH, "w", newline="") as csvfile:
        fieldnames = ["ID", "label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for row in data:
            writer.writerow(row)
            
def prepareTesting(path):
    X_test, Y_test = read_test_set(path)
    write_gold_output(Y_test)
    
def load_model():
    model = RNN()

    # Load the saved model weights
    model.load_state_dict(torch.load(RNN_PATH, map_location=torch.device('cpu')))
    model.eval()

    return model

def predict(model, X_test):
    final_predictions = []
    for i in range(len(X_test)):
        sentence = X_test[i]
        sentence = torch.tensor([char_to_index[char] for char in sentence])

        with torch.no_grad():
            predictions = model(sentence)
        
        # Get the predictions
        predictions = predictions.argmax(dim=1)
        predictions = [index_to_diacritic[index.item()] for index in predictions]
        predictions = [diacritic2id["" if diacritic == " " else diacritic] for diacritic in predictions]

        final_predictions += predictions
    
    return final_predictions

def write_output(predictions):
    data = []
    for i in range(len(predictions)):
        data.append({"ID": i, "label": predictions[i]})

    with open("./test/output.csv", "w", newline="") as csvfile:
        fieldnames = ["ID", "label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for row in data:
            writer.writerow(row)
            
def calculate_accuracy():
    output_df = pd.read_csv(OUTPUT_PATH)
    gold_output_df = pd.read_csv(GOLD_PATH)

    # Merge DataFrames on 'ID'
    merged_df = pd.merge(output_df, gold_output_df, on="ID", suffixes=('_your', '_gold'))

    # Check if the labels match
    merged_df['correct'] = merged_df['label_your'] == merged_df['label_gold']

    # Calculate accuracy
    accuracy = merged_df['correct'].mean()

    print(f"Accuracy: {accuracy * 100:.2f}%")
    
if __name__ == '__main__':
    prepareTesting(TEST_PATH)
    model = load_model()
    X_test, Y_test = read_test_set(TEST_PATH)
    predictions = predict(model, X_test)
    write_output(predictions)
    calculate_accuracy()