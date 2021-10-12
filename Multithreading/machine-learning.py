# walidacja krzyżowa
from sklearn.model_selection import StratifiedKFold
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import threading

results = []

def draw_confusion_matrix(y_test, y_pred):
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, alpha=0.3)

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()


def train(data_train, data_test, target_train, target_test):
    classifier = RandomForestClassifier(n_estimators=10)
    classifier.fit(data_train, target_train)
    y_true = target_test
    y_pred = classifier.predict(data_test)
    global results
    results.append([y_true, y_pred])

N_SPLITS = 3
SEED = 5

if __name__ == '__main__':
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    bc = datasets.load_breast_cancer()
    all_data = bc.data
    all_target = bc.target
    threads = []
    for (train_index, test_index) in kf.split(X=all_data, y=all_target):
        data_train, data_test = all_data[train_index], all_data[test_index]
        target_train, target_test = all_target[train_index], all_target[test_index]
        threads.append(threading.Thread(target=train, args=(data_train, data_test, target_train, target_test)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for result in results:
        draw_confusion_matrix(result[0], result[1])
