# walidacja krzyżowa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def train(data_train, data_test, target_train, target_test):
    classifier = RandomForestClassifier(n_estimators=10)
    classifier.fit(data_train, target_train)
    y_true = target_test
    y_pred = classifier.predict(data_test)

N_SPLITS = 5
SEED = 5

kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
all_data = [1, 2, 3]
all_target = [1, 2, 3]

for (train_index, test_index) in kf.split(X=all_data, y=all_target):
    data_train, data_test = all_data[train_index], all_data[test_index]
    target_train, target_test = all_target[train_index], all_target[test_index]
    threading.Thread(target=train, args=(data_train, data_test, target_train, target_test))

