import numpy as np
import json
from typing import Dict, List, Tuple
import os, sys
import math

AMINO_ACIDS = "ACEDGFIHKMLNQPSRTWVY"

input_file = "./data/input.jsonl"

def label_to_int(label: str) -> int:
    label_dict = {'H': 0, 'E': 1, 'C': 2}
    return label_dict[label]

def load_jsonl(filename: str)-> List[Dict]:
    with open(filename) as t:
        data = []
        ## Converts the list of dicts (jsonl file) to a single dict with id -> sequence
        for d in [json.loads(line) for line in t]:
            data.append({"sequence" : d["sequence"],
                         "label" : d["label"],
                         "resolved" : d["resolved"]})
    return data

def single_one_hot_encode(amino_acid: "str") -> np.array:
    if 0 < len(amino_acid) < 1 or amino_acid not in AMINO_ACIDS:
        raise ValueError

    # Find the index of the letter in the amino acids string
    index = AMINO_ACIDS.index(amino_acid)

    # Create a numpy array of zeros with the same length as the amino acids string
    one_hot = np.zeros(len(AMINO_ACIDS), dtype=np.int8)

    # Set the element at the index to 1
    one_hot[index] = 1

    return one_hot


def one_hot_encode_sequence(sequence:str, window_size=5) -> np.array:
    '''This function takes a sequence of amino acids and converts it into a 2D Numpy array
    representing the one-hot encoding. It has len(sequence) - 2*window_size rows and 20*(window_size*2 + 1) columns.
    '''
    num_rows = len(sequence) - 2 * window_size
    num_cols = 20 * (2 * window_size + 1)

    one_hot_sequence = np.zeros((num_rows, num_cols), dtype=np.int8)

    for i in range(num_rows):
        window = sequence[i:i + 2 * window_size + 1]
        for j, amino_acid in enumerate(window):
            one_hot = single_one_hot_encode(amino_acid)
            one_hot_sequence[i, j * 20:(j + 1) * 20] = one_hot

    return one_hot_sequence


def apply_mask(label: str, mask: str) -> List:
    if len(label) != len(mask):
        raise ValueError("Label and mask lengths do not match.")

    result = "".join(c for c, m in zip(label, mask) if m == '1')

    return [label_to_int(e) for e in result]

def one_hot_encode_labeled_sequence(entry: Dict, window_size=5) -> Tuple[np.array, np.array]:
    '''This function takes an entry dict containing the sequence, the label and the resolved information
    and returns as first component of the tuple the one-hot encoding for every residue including its sliding window that has:
    - a label
    - enough neighboring residues to fill the sliding window.
    The second component of the tuple is a Numpy array that contains the respective labels encoded as 0,1,2 for H,E,C.
    Remember: Both arrays have to have the same length; In this case internal unresolved residues should be considered
    and excluded from the encoding.
    '''
    sequence = entry['sequence']
    labels = entry['label']
    resolved = entry['resolved']

    label_array = apply_mask(labels, resolved)
    num_rows = len(sequence) - 2 * window_size
    num_cols = 20 * (2 * window_size + 1)

    one_hot_sequence = np.zeros((num_rows, num_cols), dtype=np.int8)

    for i in range(num_rows):
        window = sequence[i:i + 2 * window_size + 1]
        for j, amino_acid in enumerate(window):
            one_hot = single_one_hot_encode(amino_acid)
            one_hot_sequence[i, j * 20:(j + 1) * 20] = one_hot

    return one_hot_sequence, label_array

def predict_secondary_structure(input: np.array, labels:np.array, size_hidden=10) -> Tuple[float, float, float]:
    '''This function creates a sklearn.neural_network.MLPClassifier objects with all defaults except hidden_layer_sizes is
    set to (size_hidden,) and with random_state set to 42'''
    from sklearn.neural_network import MLPClassifier  # the neural network
    from sklearn.datasets import make_classification  # easy generation of synthetic input data
    from sklearn.model_selection import train_test_split  # to conveniantly split the data into test and training

    input_file = "./data/input.jsonl"
    entries = load_jsonl(input_file)

    X, y = make_classification(n_samples=100, random_state=1)  # 100 points, default: 20 features, 2 classes

    # the use of X for the input feature data (2D array) and y (1D) for the target values (prediction goal) is convention
    # we fix the random_state to make multiple run reproducible
    # we use stratify=y to have the same class ratios in the training and in the testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        random_state=1)

    print(X_test.shape)
    print(y_test.shape)
    # the call to fit with the provided training data is the standard way to train a model in sklearn
    clf = MLPClassifier(random_state=42, max_iter=32).fit(X_train, y_train)

    # prints out the probability for each of the classes
    # here only the first test instance is used [:1] (slicing)
    print(clf.predict_proba(X_test[:1]))

    # here we predict the class of the first 5 test instances
    print(clf.predict(X_test[:5, :]))

    # the performance on the complete test set
    print(clf.score(X_test, y_test))


def calculate_Q3(prediction: str, truth:str) -> Tuple[float,float, float]:
    '''Compares two strings of equal length:
    prediction: string of predicted states H/E/C
    truth: string of true states H/E/C
    both strings are of the same length
    returns the fraction of correct predictions for every state (H/E/C) as a 3-tuple
    '''
    if len(prediction) != len(truth):
        raise ValueError("Prediction and truth lengths do not match.")

    counts = {'H': [0, 0], 'E': [0, 0], 'C': [0, 0]}  # [correct_count, total_count]

    for p, t in zip(prediction, truth):
        if t == 'H' or t == 'E' or t == 'C':
            counts[t][1] += 1
            if p == t:
                counts[t][0] += 1

    q3_h = counts['H'][0] / counts['H'][1] if counts['H'][1] > 0 else math.nan
    q3_e = counts['E'][0] / counts['E'][1] if counts['E'][1] > 0 else math.nan
    q3_c = counts['C'][0] / counts['C'][1] if counts['C'][1] > 0 else math.nan

    return q3_h, q3_e, q3_c


if __name__ == "__main__":
    input_file = "./data/input.jsonl"
    entries = load_jsonl(input_file)
    print(len(entries))
    # extend as you need
    np.set_printoptions(threshold=sys.maxsize)

    # print(one_hot_encode_sequence("AAACYYY", window_size=2).shape)
    print(one_hot_encode_labeled_sequence({'sequence' : "AAACYYY", 'label':'HHHCEEE', 'resolved':'0010100'}, window_size=2))