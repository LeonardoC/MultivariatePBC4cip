import os
basedir = 'C:\\Users\\L03109567\\Documents\\'

datasets = os.listdir(f'{basedir}ArffDatasets\\')

results = []
avg_results = []

for ds in datasets:

    complete = True
    accs = []
    aucs = []

    for fold in range(1, 6):
        fname = basedir + f'PBCresults\\{ds}\\measures\\{ds}{fold}.csv'
        if not os.path.isfile(fname):
            complete = False
            break

        with open(fname) as f:
            f.readline()
            line = f.readline()
        parts = line.strip().split(',')
        parts = [float(i) for i in parts]
        acc, auc = parts
        accs.append(acc)
        aucs.append(auc)

    if complete:
        for i in range(5):
            results.append((ds, str(i + 1), str(accs[i]), str(aucs[i])))
        avg_acc = sum(accs) / 5
        avg_auc = sum(aucs) / 5
        avg_results.append((ds, str(avg_acc), str(avg_auc)))
    else:
        avg_results.append((ds, '0', '0'))

with open(basedir + f'PBCresults\\results.csv', 'w') as f:
    f.write('ds, fold, acc, auc\n')
    for res in results:
        f.write(','.join(res) + '\n')
with open(basedir + f'PBCresults\\avg_results.csv', 'w') as f:
    f.write('ds, acc, auc\n')
    for res in avg_results:
        f.write(','.join(res) + '\n')