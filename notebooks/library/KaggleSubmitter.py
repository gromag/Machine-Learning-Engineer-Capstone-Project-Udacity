import pandas as pd
import numpy as np

class KaggleSubmitter():

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def save_submission(data, predictions, filename = 'submission.csv'):

        assert (len(data) == len(predictions)), "Predictions and test data have different lengths"

        submission = pd.DataFrame.from_dict({
            'id': data['id'],
            'prediction': predictions.flatten()
        })

        submission.to_csv(filename, index=False)

        return submission