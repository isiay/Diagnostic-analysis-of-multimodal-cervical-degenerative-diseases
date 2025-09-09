from sklearn.metrics import hamming_loss, f1_score, jaccard_score,accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

FILTER_SCORE = 0.5

def filterOutput(output):
    filter_pred = []
    for label, score in zip(output['labels'], output['scores']):
        if score > FILTER_SCORE:
            filter_pred.append(True)
        else:
            filter_pred.append(False)
    return output['labels'][filter_pred]
        
def f1_sampled(y_true, y_pred):
    #converting the multi-label classification to a binary output
    mlb = MultiLabelBinarizer(classes=list(range(1,37)))
    gt = mlb.fit_transform(y_true)
    pred = mlb.fit_transform(y_pred)

    f1 = f1_score(gt, pred, average='samples')
    return f1

def jaccard_sampled(y_true, y_pred):
    mlb = MultiLabelBinarizer(classes=list(range(1,37)))
    gt = mlb.fit_transform(y_true)
    pred = mlb.fit_transform(y_pred)

    ja = jaccard_score(gt, pred, average='samples')
    return ja

def acc_sampled(y_true, y_pred):
    mlb = MultiLabelBinarizer(classes=list(range(1,37)))
    gt = mlb.fit_transform(y_true)
    pred = mlb.fit_transform(y_pred)

    acc = accuracy_score(gt, pred)
    return acc

def hamming_sampled(y_true, y_pred):
    mlb = MultiLabelBinarizer(classes=list(range(1,37)))
    gt = mlb.fit_transform(y_true)
    pred = mlb.fit_transform(y_pred)

    hamm = hamming_loss(gt, pred)
    return hamm