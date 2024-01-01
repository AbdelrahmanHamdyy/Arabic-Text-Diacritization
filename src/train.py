import time
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from preprocessing import *
from constants import *
from models import *

# Test GPU
device = None
device_type = None
if torch.cuda.is_available():
    device_type = "cuda"
    device = torch.device(device_type)
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print("Device name:", torch.cuda.get_device_name(0))
else:
    device_type = "cpu"
    device = torch.device(device_type)
    print("No GPU available, using the CPU instead.")
    
def data_loader(train_inputs, val_inputs, train_labels, val_labels, batch_size=BATCH_SIZE):
    # Create DataLoader for training data
    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create DataLoader for validation data
    val_data = TensorDataset(val_inputs, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader

# Specify loss function
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

def train(path, model, optimizer, train_dataloader, val_dataloader=None, epochs=NUM_EPOCHS):
    """Train the model"""

    # Tracking best validation accuracy
    best_accuracy = 0

    # Start training loop
    print("Start training...\n")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
    print("-" * 60)

    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================

        # Tracking time and loss
        t0_epoch = time.time()
        total_loss = 0

        # Put the model into the training mode
        model.train()

        for step, batch in enumerate(train_dataloader):
            # Load batch to GPU
            b_input_ids, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass
            output = model(b_input_ids)

            # Compute loss and accumulate the loss values
            loss = loss_fn(output.view(-1, output.shape[-1]), b_labels.view(-1))
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Update parameters
            optimizer.step()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        # =======================================
        #               Evaluation
        # =======================================
        if val_dataloader is not None:
            # After the completion of each training epoch, measure the model's
            # performance on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            # Track the best accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), path)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")

    print("\n")
    print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")


def evaluate(model, val_dataloader):
    """
    After the completion of each training epoch, measure the model's
    performance on our validation set.
    """
    # Put the model into the evaluation mode.
    model.eval()

    # Tracking variables
    val_loss = []
    val_accuracy = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_labels = tuple(t.to(device) for t in batch)

        # Filter out the padding value
        labels_without_pad = (b_labels != PAD)

        # Get the output
        with torch.no_grad():
            output = model(b_input_ids)

        # Compute loss
        loss = loss_fn(output.view(-1, output.shape[-1]), b_labels.view(-1))
        val_loss.append(loss.item())

        # Get the predictions
        preds = output.argmax(dim=2)

        # Calculate the accuracy
        correct_predictions = ((preds == b_labels) & labels_without_pad).sum().item()
        actual_predictions = labels_without_pad.sum().item()
        accuracy = (correct_predictions / actual_predictions) * 100

        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

def init_model(learning_rate=LEARNING_RATE):
    model = RNN()
    path = RNN_PATH

    # Send model to `device` (GPU/CPU)
    model.to(device)

    # Instantiate the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return path, model, optimizer

def prepare_data(path):
    corpus = readFile(path)

    x, y = [], []
    x_padded, y_padded = [], []

    for sentence in corpus:
        char_list, diacritics_list = separate_words_and_diacritics(sentence.strip())

        for i in range(len(char_list)):
            x.append(char_list[i])
            x.append(diacritics_list[i])

    x_tensor = [torch.tensor([char_to_index[char] for char in sentence]) for sentence in x]
    x_padded = pad_sequence(x_tensor, batch_first=True)

    y_tensor = [torch.tensor([diacritic_to_index[char] for char in sentence]) for sentence in y]
    y_padded = pad_sequence(y_tensor, batch_first=True, padding_value=PAD)
    
    return x_padded, y_padded

def execute():
    x_train_padded, y_train_padded = prepare_data(TRAIN_PATH)
    x_val_padded, y_val_padded = prepare_data(VAL_PATH)
    
    path, model, optimizer = init_model()
    print(model)
    
    train_dataloader, val_dataloader = data_loader(x_train_padded, x_val_padded, y_train_padded, y_val_padded)
    train(path, model, optimizer, train_dataloader, val_dataloader)
    
def test_model():
    x_test_padded, y_test_padded = prepare_data(TEST_PATH)

    test_data = TensorDataset(x_test_padded, y_test_padded)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

    model = RNN()
    model.load_state_dict(torch.load(RNN_PATH, map_location=torch.device(device_type)))
    model.to(device)

    loss, acc = evaluate(model, test_dataloader)

    print(f'Accuracy: {acc} | DER: {1 - (acc / 100)}\n')
    
if __name__ == '__main__':
    execute()
    test_model()