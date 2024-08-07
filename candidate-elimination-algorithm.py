import csv

def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

def more_general(h1, h2):
    """Return True if h1 is more general than or equal to h2"""
    more_general_parts = []
    for x, y in zip(h1, h2):
        mg = x == "?" or (x != "0" and (x == y or y == "0"))
        more_general_parts.append(mg)
    return all(more_general_parts)

def fulfills(example, hypothesis):
    """Return True if hypothesis fulfills example"""
    return more_general(hypothesis, example)

def min_generalizations(h, x):
    """Return a list of minimal generalizations of h that fulfill x"""
    h_new = list(h)
    for i in range(len(h)):
        if not fulfills(x[i:i+1], h[i:i+1]):
            h_new[i] = '?' if h[i] != "0" else x[i]
    return [tuple(h_new)]

def min_specializations(h, domains, x):
    """Return a list of minimal specializations of h that fail x"""
    results = []
    for i in range(len(h)):
        if h[i] == "?":
            for val in domains[i]:
                if x[i] != val:
                    h_new = h[:i] + (val,) + h[i+1:]
                    results.append(h_new)
        elif h[i] != "0":
            h_new = h[:i] + ("0",) + h[i+1:]
            results.append(h_new)
    return results

def candidate_elimination(training_data):
    domains = [list(set([x[i] for x in training_data])) for i in range(len(training_data[0])-1)]
    G = {("?",) * (len(training_data[0]) - 1)}
    S = {("0",) * (len(training_data[0]) - 1)}
    for instance in training_data:
        x, label = instance[:-1], instance[-1]
        if label == "Yes": # positive example
            G = {g for g in G if fulfills(x, g)}
            S = generalize_S(x, S, G)
        else: # negative example
            S = {s for s in S if not fulfills(x, s)}
            G = specialize_G(x, G, S, domains)
        G = {g for g in G if any([more_general(g, s) for s in S])}
        S = {s for s in S if any([more_general(g, s) for g in G])}
    return S, G

def generalize_S(x, S, G):
    S_prev = list(S)
    for s in S_prev:
        if not fulfills(x, s):
            S.remove(s)
            Splus = min_generalizations(s, x)
            S.update([h for h in Splus if any([more_general(g, h) for g in G])])
    return S

def specialize_G(x, G, S, domains):
    G_prev = list(G)
    for g in G_prev:
        if fulfills(x, g):
            G.remove(g)
            Gminus = min_specializations(g, domains, x)
            G.update([h for h in Gminus if any([more_general(h, s) for s in S])])
    return G

# Example usage:
file_path = '../Sample Datasets for project/training_data.csv'
training_data = read_csv(file_path)
S, G = candidate_elimination(training_data)

print("S (Specific boundary):", S)
print("G (General boundary):", G)
