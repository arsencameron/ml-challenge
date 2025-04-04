import sys
import csv
import random
import numpy as np
import pandas as pd

# Adjust these to match your dataset columns and final class labels
column_rename = {
    "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)": "1",
    "Q2: How many ingredients would you expect this food item to contain?": "2",
    "Q3: In what setting would you expect this food to be served? Please check all that apply": "3",
    "Q4: How much would you expect to pay for one serving of this food item?": "4",
    "Q5: What movie do you think of when thinking of this food item?": "5",
    "Q6: What drink would you pair with this food item?": "6",
    "Q7: When you think about this food item, who does it remind you of?": "7",
    "Q8: How much hot sauce would you add to this food item?": "8",
    "Label": "label"
}

class_map = {
    0: "Pizza",
    1: "Shawarma",
    2: "Sushi"
}

def load_vocabularies():
    data = np.load("vocabularies.npz", allow_pickle=True)
    vocab_q2 = data["vocab_q2"].tolist()
    vocab_q4 = data["vocab_q4"].tolist()
    vocab_q5 = data["vocab_q5"].tolist()
    vocab_q6 = data["vocab_q6"].tolist()
    q3_cols = data["expected_q3_columns"].tolist()
    q7_cols = data["expected_q7_columns"].tolist()
    return vocab_q2, vocab_q4, vocab_q5, vocab_q6, q3_cols, q7_cols

def make_bow(series, vocab):
    length = len(vocab)
    X = np.zeros((len(series), length), dtype=int)
    vocab_dict = {word: idx for idx, word in enumerate(vocab)}
    for i, entry in enumerate(series):
        words = set(str(entry).lower().split())
        for w in words:
            if w in vocab_dict:
                X[i, vocab_dict[w]] = 1
    return X

def build_features(df, vocab_q2, vocab_q4, vocab_q5, vocab_q6, q3_cols, q7_cols):
    if "label" in df.columns:
        df = df.drop(columns=["label"])
    df["1"] = df["1"].astype(int)
    X_q2 = make_bow(df["2"], vocab_q2)
    X_q4 = make_bow(df["4"], vocab_q4)
    X_q5 = make_bow(df["5"], vocab_q5)
    X_q6 = make_bow(df["6"], vocab_q6)

    X_q3 = df["3"].str.get_dummies(sep=",")
    X_q3 = X_q3.reindex(columns=q3_cols, fill_value=0)

    X_q7 = df["7"].str.get_dummies(sep=",")
    X_q7 = X_q7.reindex(columns=q7_cols, fill_value=0)

    df["8"] = df["8"].astype("category").cat.codes
    return np.hstack([
        df["1"].values.reshape(-1, 1),
        X_q2,
        X_q3.values,
        X_q4,
        X_q5,
        X_q6,
        X_q7.values,
        df["8"].values.reshape(-1, 1)
    ])

def load_mlp_preproc():
    param_data = np.load("mlp_params.npz", allow_pickle=True)
    non_constant_cols = param_data["non_constant_columns"]
    X_mean = param_data["X_mean"]
    X_std = param_data["X_std"]
    return non_constant_cols, X_mean, X_std

def load_mlp_model():
    d = np.load("mlp_model.npz", allow_pickle=True)
    weights = d["weights"]
    biases = d["biases"]
    return weights, biases

def mlp_predict(X, weights_list, biases_list):
    A = X
    for i in range(len(weights_list) - 1):
        Z = A @ weights_list[i] + biases_list[i]
        A = np.maximum(0, Z)
    Z_out = A @ weights_list[-1] + biases_list[-1]
    exp_vals = np.exp(Z_out - np.max(Z_out, axis=1, keepdims=True))
    probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

def predict_all(filename):
    df = pd.read_csv(filename)
    df = df.rename(columns=column_rename)

    vocab_q2, vocab_q4, vocab_q5, vocab_q6, q3_cols, q7_cols = load_vocabularies()
    X_all = build_features(df, vocab_q2, vocab_q4, vocab_q5, vocab_q6, q3_cols, q7_cols)

    non_constant_cols, X_mean, X_std = load_mlp_preproc()
    X_all = X_all[:, non_constant_cols]
    X_all = np.nan_to_num(X_all)
    eps = 1e-8
    X_all = (X_all - X_mean) / (X_std + eps)

    w_list, b_list = load_mlp_model()
    preds = mlp_predict(X_all, w_list, b_list)
    result = [class_map.get(p, "Pizza") for p in preds]
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        csv_file = "unseen_data.csv"
    else:
        csv_file = sys.argv[1]
    output_preds = predict_all(csv_file)
    for op in output_preds:
        print(op)
