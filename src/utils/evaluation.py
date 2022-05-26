from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


def get_classification_scores(event_sequences: List[List[Tuple]]) -> Dict:
    true_labels: List[int] = []
    predicted_labels: List[int] = []

    for event_sequence in event_sequences:
        for event in event_sequence:
            true_labels.append(event[1])
            predicted_labels.append(np.argmax(event[2]))

    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "f1_score_weighted": f1_score(true_labels, predicted_labels, average="weighted"),
        "classification_report": pd.DataFrame(
            classification_report(true_labels, predicted_labels, output_dict=True)
        ).transpose(),
    }
