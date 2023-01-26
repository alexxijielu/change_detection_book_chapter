import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv
import umap
import sklearn.metrics.pairwise as skdist

def visualize_global_effects(control_features, condition_features, protein_names, labels=[], scale=True):
    combined_features = np.concatenate((control_features, condition_features))

    if scale:
        scaler = StandardScaler()
        scaler.fit(combined_features)
        combined_features = scaler.transform(combined_features)
        control_features = scaler.transform(control_features)
        condition_features = scaler.transform(condition_features)
    if len(labels) != 0:
        classes = np.unique(labels)

    encoder = umap.UMAP(metric="euclidean", n_neighbors=30, spread=0.1, random_state=42).fit(combined_features)
    control_projection = encoder.transform(control_features)
    condition_projection = encoder.transform(condition_features)

    gridsize = 20
    grid = np.zeros((gridsize, gridsize, 2))
    xmin = np.min(control_projection[:, 0])
    xmax = np.max(control_projection[:, 0])
    ymin = np.min(control_projection[:, 1])
    ymax = np.max(control_projection[:, 1])
    x_increments = (xmax - xmin) / gridsize
    y_increments = (ymax - ymin) / gridsize

    for xg in range (0, gridsize):
        for yg in range(0, gridsize):
            curr_grid = (xmin + xg * x_increments,
                         xmin + (xg + 1) * x_increments,
                         ymin + yg * y_increments,
                         ymin + (yg + 1) * y_increments)
            datapoints = np.where((control_projection[:, 0] > curr_grid[0]) &
                                  (control_projection[:, 0] < curr_grid[1]) &
                                  (control_projection[:, 1] > curr_grid[2]) &
                                  (control_projection[:, 1] < curr_grid[3]))[0]
            if len(datapoints) >= 20:
                differences = control_projection[datapoints] - condition_projection[datapoints]
                grid[xg, yg] = np.average(differences, axis=0)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot()
    i = 0

    if len(labels) != 0:
        cmap = plt.get_cmap('tab20')
        c_ids = cmap(np.linspace(0, 1.0, len(classes)))
        for c in classes:
            ix = np.where(labels == c)
            ax.scatter(control_projection[ix, 0], control_projection[ix, 1], c=np.array([c_ids[i]]),
                       marker='o', label=c, s=3)
            i += 1
    else:
        ax.scatter(control_projection[:, 0], control_projection[:, 1], marker='o', s=3)

    for xg in range (0, gridsize):
        for yg in range(0, gridsize):
            if grid[xg, yg, 0] != 0 or grid[xg, yg, 1] != 0:
                xcen = xmin + xg * x_increments + x_increments / 2
                ycen = ymin + yg * y_increments + y_increments / 2
                ax.arrow(xcen, ycen, grid[xg, yg, 0], grid[xg, yg, 1],
                         length_includes_head=True, head_width=0.01, head_length=0.01, color="black")

    ax.set_xticks([])
    ax.set_yticks([])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=3)
    plt.show()


def visualize_changes(control_features, condition_features, change_matrix, protein_names,
                      labels=[], change_list=np.arange(0, 10), scale=True):
    combined_features = np.concatenate((control_features, condition_features))
    change_ranking = np.argsort(np.sum(np.abs(change_matrix), axis=1))[::-1][change_list]

    if scale:
        scaler = StandardScaler()
        scaler.fit(combined_features)
        combined_features = scaler.transform(combined_features)
        control_features = scaler.transform(control_features)
        condition_features = scaler.transform(condition_features)
    if len(labels) != 0:
        classes = np.unique(labels)

    encoder = umap.UMAP(metric="euclidean", n_neighbors=30, spread=0.1, random_state=42).fit(combined_features)
    control_projection = encoder.transform(control_features)
    condition_projection = encoder.transform(condition_features)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot()
    i = 0
    if len(labels) != 0:
        cmap = plt.get_cmap('tab20')
        c_ids = cmap(np.linspace(0, 1.0, len(classes)))
        for c in classes:
            ix = np.where(labels == c)
            ax.scatter(control_projection[ix, 0], control_projection[ix, 1], color=np.array([c_ids[i]]),
                       marker='o', label=c, s=3)
            i += 1
    else:
        ax.scatter(control_projection[:, 0], control_projection[:, 1], marker='o', s=3)

    dist = skdist.pairwise_distances(control_features, metric="euclidean")
    nearest = np.argsort(dist, axis=1)[:, 1:51]

    for pidx in change_ranking:
        pname = protein_names[pidx]
        neighbors = nearest[pidx]
        if len(labels) != 0:
            plabel = labels[pidx]
            plabelidx = np.where(classes == plabel)[0][0]

        neighbor_change = np.average(condition_projection[neighbors] - control_projection[neighbors], axis=0)
        corrected_change = (condition_projection[pidx, :] - control_projection[pidx, :]) - neighbor_change

        text_adj = [0, 0]
        if corrected_change[0] > 0:
            text_adj[0] = -0.02
        if corrected_change[0] < 0:
            text_adj[0] = 0.02
        if corrected_change[1] > 0:
            text_adj[1] = -0.02
        if corrected_change[1] < 0:
            text_adj[1] = 0.02

        txt = ax.text(control_projection[pidx, 0] + text_adj[0],
                control_projection[pidx, 1] + text_adj[1],
                pname, fontsize=8, fontweight = 'bold', horizontalalignment='center', verticalalignment='center')
        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])

        if len(labels) != 0:
            ax.scatter(control_projection[pidx, 0], control_projection[pidx, 1],
                       color=c_ids[plabelidx], edgecolors='black', marker='o', linewidth=1, s=50, alpha=0.7)

        ax.arrow(control_projection[pidx, 0],
                 control_projection[pidx, 1],
                 corrected_change[0],
                 corrected_change[1],
                 length_includes_head=True, head_width=0.02, head_length=0.02, color="black", alpha=1.0, linewidth=0.2)

    ax.set_xticks([])
    ax.set_yticks([])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=3)
    plt.show()

def filter_matrices (reference, condition):
    '''Preprocessing operation - calculates the intersection of proteins between the reference and the
    condition, and sorts them so that they're in the same order.'''
    ref_headers, ref_genelist, ref_genematrix = open_matrix(reference)
    cond_headers, cond_genelist, cond_genematrix = open_matrix(condition)

    sorted_ref = []
    sorted_cond = []
    # Get the intersection of the list
    # Sometimes there can be duplicate proteins, so we'll just take the first occurrence if there is
    intersect = np.intersect1d(ref_genelist, cond_genelist)
    for protein in intersect:
        ref_index = np.where(ref_genelist == protein)[0][0]
        cond_index = np.where(cond_genelist == protein)[0][0]
        sorted_ref.append(ref_genematrix[ref_index])
        sorted_cond.append(cond_genematrix[cond_index])

    sorted_ref = np.array(sorted_ref)
    sorted_cond = np.array(sorted_cond)

    return intersect, sorted_ref, sorted_cond, cond_headers


def open_matrix(fileName):
    '''Opens a gene matrix file and returns the feature labels, gene labels and gene matrix
    Input: Path of file to be opened (as a string)
    Output: feature labels, gene labels, and gene matrix'''
    file = open(fileName)
    list = csv.reader(file, delimiter='\t')
    matrix = np.array([row for row in list])

    genelist = matrix[1:, 0]
    headers = matrix[0, :]
    genematrix = matrix[1:, 1:]
    try:
        genematrix = genematrix.astype(float)
    except:
        pass

    file.close()
    return headers, genelist, genematrix


def get_localization_labels(protein_list, conversion):
    conversion = np.array([x for x in csv.reader(open(conversion), delimiter="\t")])
    localizations = []
    for p in protein_list:
        try:
            localizations.append(conversion[np.where(conversion[:, 0] == p)][0][2])
        except IndexError:
            localizations.append("unknown")
    return np.array(localizations)

def clean_localization_labels(localization_labels):
    for i in range(0, len(localization_labels)):
        if "," in localization_labels[i] or localization_labels[i] == "ambiguous" or \
                localization_labels[i] == "unknown":
            localization_labels[i] = "multi-localizing/unknown"
        elif localization_labels[i] == "early Golgi" or localization_labels[i] == "late Golgi" \
                or localization_labels[i] == "ER to Golgi":
            localization_labels[i] = "Golgi"
        elif localization_labels[i] == "punctate composite" or localization_labels[i] == "actin" \
            or localization_labels[i] == "lipid particle" or localization_labels[i] == "endosome" \
            or localization_labels[i] == "peroxisome" or localization_labels[i] == "spindle pole":
                localization_labels[i] = "punctate"
    return localization_labels


if __name__ == "__main__":
    control = "../data/HOwt_features.tsv"
    condition = "../data/HU02_features.tsv"
    change = "../data/HU02_changes.tsv"

    protein_names, sorted_control, sorted_condition, _ = filter_matrices(control, condition)
    _, _, change = open_matrix(change)
    localization_labels = get_localization_labels(protein_names, '../data/yeast_localizations.tsv')

    for i in range(0, len(localization_labels)):
        if "," in localization_labels[i] or localization_labels[i] == "ambiguous" or \
                localization_labels[i] == "unknown":
            localization_labels[i] = "multi-localizing/unknown"
        elif localization_labels[i] == "early Golgi" or localization_labels[i] == "late Golgi" \
                or localization_labels[i] == "ER to Golgi":
            localization_labels[i] = "Golgi"
        elif localization_labels[i] == "punctate composite" or localization_labels[i] == "actin" \
            or localization_labels[i] == "lipid particle" or localization_labels[i] == "endosome" \
            or localization_labels[i] == "peroxisome" or localization_labels[i] == "spindle pole":
                localization_labels[i] = "punctate"

    #visualize_global_effects(sorted_control, sorted_condition, protein_names, localization_labels)
    visualize_changes(sorted_control, sorted_condition, change, protein_names, localization_labels,
                      change_list=np.arange(0, 10), scale=True)
