from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn.svm import SVC


def print_results(labels_vec, predictions_vec):
    tn, fp, fn, tp = confusion_matrix(labels_vec, predictions_vec).ravel()
    f1_performance = f1_score(labels_vec, predictions_vec, average='weighted')
    # f1_performance = f1_score(labels_vec, predictions_vec, pos_label=1)
    acc_performance = accuracy_score(labels_vec, predictions_vec)
    mcc_performance = matthews_corrcoef(labels_vec, predictions_vec)
    print("tn, fp, fn, tp: ", tn, fp, fn, tp)
    print("f1_performance: ", f1_performance)
    print("acc_performance", acc_performance)
    print("mcc_performance: ", mcc_performance)


def evaluation(train_set, test_set):

    clf_ert = ExtraTreesClassifier(n_estimators=200)  # n_estimators=100
    clf_rf = RandomForestClassifier(n_estimators=200)  # n_estimators=100
    clf_xgb = GradientBoostingClassifier(n_estimators=200)  # n_estimators=100
    clf_dt = DecisionTreeClassifier()
    clf_ls = make_pipeline(StandardScaler(), LinearSVC(tol=1e-5))
    clf_rbfsvm = SVC(kernel='rbf', gamma=1)

    clf_ert.fit(train_set[:, 0:-1], train_set[:, -1])
    clf_rf.fit(train_set[:, 0:-1], train_set[:, -1])
    clf_xgb.fit(train_set[:, 0:-1], train_set[:, -1])
    clf_dt.fit(train_set[:, 0:-1], train_set[:, -1])
    clf_ls.fit(train_set[:, 0:-1], train_set[:, -1])
    clf_rbfsvm.fit(train_set[:, 0:-1], train_set[:, -1])

    prediction_ert = clf_ert.predict(test_set[:, 0:-1])
    prediction_rf = clf_rf.predict(test_set[:, 0:-1])
    prediction_xgb = clf_xgb.predict(test_set[:, 0:-1])
    prediction_dt = clf_dt.predict(test_set[:, 0:-1])
    prediction_ls = clf_ls.predict(test_set[:, 0:-1])
    prediction_rbfsvm = clf_ls.predict(test_set[:, 0:-1])

    groundTruth_vec = test_set[:, -1]

    return groundTruth_vec, prediction_ert, prediction_rf, prediction_xgb, prediction_dt, prediction_ls, prediction_rbfsvm



