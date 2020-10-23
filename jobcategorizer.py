import re
import json
import numpy as np
import pandas as pd
import tensorflow as tf

def load_models():
    global tokenizer, model, ord_categories

    # loading tokenizer
    with open(file='tokenizer.json', mode='r') as f:
        data = json.load(fp=f)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string=data)

    # loading trained model
    model = tf.keras.models.load_model(filepath='model.h5')

    # loading categories
    ord_categories = []
    with open(file='categories.txt', mode='r') as f:
        for line in f:
            ord_categories.append(line[:-1])


def find_category(txt):
    fnd_pttrn = re.search(pattern='[tsop|noitisop|elor](.+?)rof', string=txt[::-1])
    if fnd_pttrn:
        test_txt = ' '.join(fnd_pttrn.group()[::-1].strip().split()[1:-1])
        test_seq = tokenizer.texts_to_sequences(texts=[test_txt])
        test_padded_seq = tf.keras.preprocessing.sequence.pad_sequences(test_seq, maxlen=MX_LN, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
        prediction = np.argmax(model.predict(x=test_padded_seq), axis=-1)
        return (test_txt, pd.Categorical.from_codes(codes=prediction, categories=ord_categories)[0])

    else:
        return (None, 'No role found!')