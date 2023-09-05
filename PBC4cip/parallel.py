import os
from pathlib import Path
from multiprocessing import Pool, freeze_support
import numpy as np
import pandas as pd

from core.PBC4cip import PBC4cip
from core.Evaluation import obtainAUCMulticlass
from core.Helpers import get_col_dist, get_idx_val
from core.Dataset import PandasDataset
from core.DecisionTreeBuilder import DecisionTreeBuilder, MultivariateDecisionTreeBuilder
from core.SupervisedClassifier import DecisionTreeClassifier
from core.DistributionEvaluatorHelper import get_distribution_evaluator
from core.EmergingPatterns import EmergingPatternCreator,EmergingPatternSimplifier
from core.DistributionTester import AlwaysTrue
from core.Item import ItemComparer, SubsetRelation

from scipy.io.arff import loadarff

#basedir = 'C:\\Users\\L03109567\\Documents\\'
basedir = 'D:\\Leo\\'

def import_data(trainFile, testFile):
    if trainFile.endswith('csv'):
        train = pd.read_csv(trainFile) 
    elif trainFile.endswith('arff'):
        train = import_arff(trainFile)
    if testFile.endswith('csv'):
        test = pd.read_csv(testFile)
    elif testFile.endswith('arff'):
        test = import_arff(testFile)
    return train, test

def import_arff(filename):
    df = loadarff(filename)
    return pd.DataFrame(df[0])

def split_data(train, test):
    X_train = train.iloc[:,  0:train.shape[1]-1]
    y_train =  train.iloc[:, train.shape[1]-1 : train.shape[1]]

    X_test = test.iloc[:,  0:test.shape[1]-1]
    y_test =  test.iloc[:, test.shape[1]-1 : test.shape[1]]

    return X_train, y_train, X_test, y_test

def score(predicted, y):
        y_class_dist = get_col_dist(y[f'{y.columns[0]}'])
        real = list(map(lambda instance: get_idx_val(y_class_dist, instance), y[f'{y.columns[0]}']))
        numClasses = len(y_class_dist)
        confusion = [[0]* numClasses for i in range(numClasses)]
        classified_as = 0
        error_count = 0

        for i in range(len(real)):
            if real[i] != predicted[i]:
                error_count = error_count + 1
            confusion[real[i]][predicted[i]] = confusion[real[i]][predicted[i]] + 1

        acc = 100.0 * (len(real) - error_count) / len(real)
        auc = obtainAUCMulticlass(confusion, numClasses)

        return confusion, acc, auc

def test_tree(trainFile, testFile):
    train, test = import_data(trainFile, testFile)
    X_train, y_train, X_test, y_test = split_data(train, test)
    dataset = PandasDataset(X_train,y_train)
    
    X_train = X_train.to_numpy()
    #y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()


    decisionTreeBuilder = MultivariateDecisionTreeBuilder(dataset, X_train, y_train.to_numpy())
    decisionTreeBuilder.distributionEvaluator = get_distribution_evaluator('quinlan gain')
    decisionTreeBuilder.FeatureCount = len(dataset.Attributes)
    #def SampleWithoutRepetition(population, sample_size):
    decisionTreeBuilder.OnSelectingFeaturesToConsider = lambda pop, size: pop
    decisionTree = decisionTreeBuilder.Build()
    classifier = DecisionTreeClassifier(decisionTree)
    print(decisionTree.TreeRootNode)
    y_pred = classifier.predict(X_test)

    confusion, acc, auc = score(y_pred, y_test)

    print(f"\nConfusion Matrix:")
    for i in range(len(confusion[0])):
        for j in range(len(confusion[0])):
            print(f"{confusion[i][j]} ", end='')
        print("")
    print(f"\n\nacc: {acc} , auc: {auc}")

def test_miner(trainFile, testFile):

    train, test = import_data(trainFile, testFile)
    X_train, y_train, X_test, y_test = split_data(train, test)
    dataset = PandasDataset(X_train,y_train)
    


    decisionTreeBuilder = MultivariateDecisionTreeBuilder(dataset, X_train.to_numpy(), y_train.to_numpy())
    decisionTreeBuilder.distributionEvaluator = get_distribution_evaluator('quinlan gain')
    decisionTreeBuilder.FeatureCount = len(dataset.Attributes)
    #def SampleWithoutRepetition(population, sample_size):
    decisionTreeBuilder.OnSelectingFeaturesToConsider = lambda pop, _: pop
    decisionTree = decisionTreeBuilder.Build()

    print(decisionTree.TreeRootNode)

    classifier = DecisionTreeClassifier(decisionTree)
    dataset = PandasDataset(X_train,y_train)
    epCreator = EmergingPatternCreator(dataset)
    PatternsList = []

    simplifier = EmergingPatternSimplifier(ItemComparer().Compare)
    def PatternFound(pattern):        
        if AlwaysTrue(pattern.Counts, dataset.Model, dataset.Class):
            simplifiedPattern = simplifier.Simplify(pattern)
            PatternsList.append(simplifiedPattern)

    epCreator.ExtractPatterns(classifier, PatternFound)
    print("PATTERNS:\n")
    for pattern in PatternsList:
        print(pattern)
    print(len(PatternsList))

    
def test_PBC4cip(trainFile, testFile):
    train, test = import_data(trainFile, testFile)
    X_train, y_train, X_test, y_test = split_data(train, test)
    
    classifier = PBC4cip(multivariate=True, distribution_evaluator='twoing', tree_count= 200, filtering= False)
    patterns = classifier.fit(X_train, y_train)

    # y_test_scores = classifier.score_samples(X_test)
    # print("Test Scores:")
    # for i, test_score in enumerate(y_test_scores):
    #     print(f"{i}: {test_score}")
    
    y_pred = classifier.predict(X_test)
    confusion, acc, auc = score(y_pred, y_test)

    # print(f"\nPatterns Found:")
    # patterns = sorted(patterns, key= lambda x: max(x.Supports))
    # for pattern in patterns:
    #     print(f"{pattern}")
    
    # print(len(patterns))

    # print(f"\nConfusion Matrix:")
    # for i in range(len(confusion[0])):
    #     for j in range(len(confusion[0])):
    #         print(f"{confusion[i][j]} ", end='')
    #     print("")
    # print(f"\n\nacc: {acc} , auc: {auc} , numPatterns: {len(patterns)}")
    return confusion, acc, auc    





def runPBC4cip(basedir, ds, fold):
    trainFile = f'{basedir}ArffDatasets\\{ds}\\{ds}{fold}tra.arff'
    testFile = f'{basedir}ArffDatasets\\{ds}\\{ds}{fold}tst.arff'
    confusion, acc, auc = test_PBC4cip(trainFile, testFile)
    
    confusionDir = f'{basedir}PBCresults\\{ds}\\confusion\\'
    measuresDir =f'{basedir}PBCresults\\{ds}\\measures\\'


    with open(f'{confusionDir}{ds}{fold}.csv', 'w') as f:
        for row in confusion:
            f.write(','.join([str(i) for i in row]) + '\n')

    with open(f'{measuresDir}{ds}{fold}.csv', 'w') as f:
        f.write('acc, auc\n')
        f.write(f'{acc}, {auc}\n')

    print(f"{ds}{fold}. acc: {acc} , auc: {auc}")

def calculate(func, args):
    func(*args)

def runPBC4cipParallel():
    freeze_support()
    datasets = os.listdir(f'{basedir}ArffDatasets\\')
    #datasets = ['iris']
    with Pool(processes=20) as pool:
        TASKS = []
        for ds in datasets:
            confusionDir = f'{basedir}PBCresults\\{ds}\\confusion\\'
            Path(confusionDir).mkdir(parents= True, exist_ok=True)
            measuresDir =f'{basedir}PBCresults\\{ds}\\measures\\'
            Path(measuresDir).mkdir(parents= True, exist_ok=True)
            for fold in range(1, 6):
                fname = measuresDir + ds + str(fold) + ".csv"
                if not os.path.isfile(fname):
                    TASKS.append((runPBC4cip, (basedir, ds, fold)))
        results = [pool.apply_async(calculate, t) for t in TASKS]
        
        for r in results:
            r.get()

if __name__ == "__main__":
    runPBC4cipParallel()
    #trainFile = current_location + '\\example\\train.csv'
    #testFile = current_location + '\\example\\test.csv'
    
    #test_miner(trainFile, testFile)

    #test_tree(trainFile, testFile)
    #print(type(train[0][0]))
    #print(train[0]['sepallength'])