datasets = ['ACIC 2016', 'ACIC 2017', 'IHDP', 'JOBS', 'NEWS', 'TWINS']
for dataset in datasets:
    filename = f'saved/variance_{dataset}.csv'
    filename = 'saved/variance_by_entropy_RORCO.csv'

    lines = []

    undesirable = ['nan']

    with open(filename, 'r') as f:
        for line in f.readlines():
            # Remove all nan entries
            new_line = []
            for part in line.split('), '):
                if not ('Double-Double' in line and 'Doubly Robust' in part):
                    new_line.append(part)
            new_line = '), '.join(new_line).replace('\n', '')
            if new_line[0] != '{':
                new_line = '{' + new_line
            if new_line[-1] != '}':
                new_line = new_line + ')}'
            lines.append(new_line)

    with open(filename, 'w') as f:
        for line in lines:
            f.write(line + '\n')