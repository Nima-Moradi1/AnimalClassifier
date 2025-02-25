import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    # اینجا دیتای تست رو لود میکنیم و میدیم به مدل
    X_test = np.load("scripts/X_test.npy")
    y_test = np.load("scripts/y_test.npy")
    model = load_model("models/animal_classifier.keras")

    # فرایند پریدکت رو اینجا با ارگمکس انجام میدیم
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    print(classification_report(y_test, y_pred, target_names=["Cat", "Dog", "Fox"]))

    # ماتریکس مهم پلات که بر اساس پردازش تصویر دیتای ما رو بهمون میده
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Cat", "Dog", "Fox"], yticklabels=["Cat", "Dog", "Fox"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()