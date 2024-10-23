import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Assuming you have predicted probabilities and true labels
y_true = [0.42, 0.33, 0.25, 0.2, 0.42]
y_scores = [0.75, 0.77, 0.72, 0.8, 0.75]

fpr, tpr, thresholds = roc_curve(y_true, y_scores)

plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic   (ROC) Curve')
plt.show()