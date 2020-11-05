import pandas as pd
import numpy as np


def calculate_score(probs):
    pdo = 20
    odd = 50
    odds = (1 - probs) / probs
    factor = pdo / np.log(2)
    offset = 800 - factor * np.log(odd)
    score = [offset + factor * np.log(i) for i in odds]
    return score


if __name__ == "__main__":
    """
    This block is for model deployment! To be updated...
    """
    score = calculate_score(probs)
    print("Score of the given customer is {}".format(score))
