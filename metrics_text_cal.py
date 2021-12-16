import tensorflow as tf
import sys
from datetime import datetime

input_test_filename = sys.argv[1]
output_score_filename = "scores.txt"

pred = []
y_true = []
pred_acc = []
with open(input_test_filename, 'r') as input:
    next(input)
    for line in input:
        line_list = line.strip().split(",")
        pred_item = float(line_list[1])
        pred_acc_item = int(round(pred_item))
        true_item = int(line_list[-1])
        pred.append(pred_item)
        y_true.append(true_item)
        pred_acc.append(pred_acc_item)


assert len(y_true) == len(pred)

auroc = tf.keras.metrics.AUC(curve='ROC')
auroc.update_state(y_pred=pred,y_true=y_true)
auroc_result = auroc.result().numpy()

auprc = tf.keras.metrics.AUC(curve='PR')
auprc.update_state(y_pred=pred,y_true=y_true)
auprc_result = auprc.result().numpy()

accuracy = tf.keras.metrics.Accuracy()
accuracy.update_state(y_pred=pred_acc,y_true=y_true)
accuracy_result = accuracy.result().numpy()

TN = tf.keras.metrics.TrueNegatives()
TN.update_state(y_pred=pred,y_true=y_true)
TN_result = TN.result().numpy()

TP = tf.keras.metrics.TruePositives()
TP.update_state(y_pred=pred,y_true=y_true)
TP_result = TP.result().numpy()

FN = tf.keras.metrics.FalseNegatives()
FN.update_state(y_pred=pred,y_true=y_true)
FN_result = FN.result().numpy()

FP = tf.keras.metrics.FalsePositives()
FP.update_state(y_pred=pred,y_true=y_true)
FP_result = FP.result().numpy()

mean_squared_error = tf.keras.metrics.MeanSquaredError()
mean_squared_error.update_state(y_pred=pred,y_true=y_true)
mean_squared_error_result = mean_squared_error.result().numpy()

precision = tf.keras.metrics.Precision()
precision.update_state(y_pred=pred,y_true=y_true)
precision_result = precision.result().numpy()

recall = tf.keras.metrics.Recall()
recall.update_state(y_pred=pred,y_true=y_true)
recall_result = recall.result().numpy()

def fscore(beta, p, r):
    f_result = (1+beta*beta)*p*r/(beta*beta*p + r)
    return f_result

F1score = fscore(1, precision_result, recall_result)
Fscore_recall = fscore(2, precision_result, recall_result)
Fscore_precision = fscore(0.5, precision_result, recall_result)

with open(output_score_filename,'w') as output:
    output.write("{}=================\n".format(datetime.now()))
    output.write("AUROC: {}\n".format(auroc_result))
    output.write("AUPRC: {}\n".format(auprc_result))
    output.write("F1score: {}\n".format(F1score))
    output.write("Accuracy: {}\n".format(accuracy_result))
    output.write("\n")
    output.write("MeanSquaredError: {}\n".format(mean_squared_error_result))
    output.write("Precision: {}\n".format(precision_result))
    output.write("Recall: {}\n".format(recall_result))
    output.write("Fscore_recall: {}\n".format(Fscore_recall))
    output.write("Fscore_precision: {}\n".format(Fscore_precision))
    output.write("TruePositive: {}\n".format(int(TP_result)))
    output.write("TrueNegative: {}\n".format(int(TN_result)))
    output.write("FalsePositive: {}\n".format(int(FP_result)))
    output.write("FalseNegative: {}\n".format(int(FN_result)))


