from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

##############################################################################################################
##############################################################################################################

accuracy = lambda actual, predictions: accuracy_score(actual, predictions)
precision = lambda actual, predictions: precision_score(actual, predictions, average='binary')
recall = lambda actual, predictions: recall_score(actual, predictions, average='binary')
f_score = lambda actual, predictions: f1_score(actual, predictions, average='binary')
roc_auc = lambda actual, predictions: roc_auc_score(actual, predictions)

##############################################################################################################
##############################################################################################################

def confusion_matrix(actual, predicted):
    conf_matrix = confusion_matrix(actual, predicted)
    tn, fp, fn, tp = conf_matrix.ravel()
    return tn / (tn + fp)