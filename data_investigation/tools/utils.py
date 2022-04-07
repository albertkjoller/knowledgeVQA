import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import json

from pathlib import Path
from tqdm.notebook import tqdm
from itertools import accumulate
from collections import Counter

import plotly.graph_objects as go

import dash
from dash import dcc
from dash import html


# data load function
def loadData(filename, data_path):
    filetype = filename.split(".")[-1]
    if filetype == 'npy':
        data = np.load(data_path / filename, allow_pickle=True)
        dataset_specs = data[0]
        data = data[1:]
        q_type = None
        
    elif filetype == 'json':
        data = json.load(open(data_path/filename))
        keys = set(data.keys())
        keys.discard('annotations')
        q_type = data['question_types']
        dataset_specs = {key: data[key] for key in keys}    
        data = pd.DataFrame(data['annotations'])
        
    
    print("Dataset specs:")
    for key, value in dataset_specs.items():
        print(f"{key}: {value}")
    
    return pd.DataFrame.from_records(data), q_type


def plot_PCA(fig, PCA_df, attribute, subplot=111, chosen_PCs=(1,2)):
    # create subplot
    ax = fig.add_subplot(subplot) 

    # add labels
    ax.set_xlabel(f'Principal Component {chosen_PCs[0]}', fontsize = 15)
    ax.set_ylabel(f'Principal Component {chosen_PCs[1]}', fontsize = 15)
    ax.set_title(f'PCA ({attribute})', fontsize = 20)

    # specify colors and targets
    targets = sorted(list(PCA_df[attribute].unique()))
    colors = [f'C{i}' for i in range(1, 1+targets.__len__())]

    # scatter plot colored by the answer_variability
    for target, color in zip(targets, colors):
        idxs = PCA_df[attribute] == target
        ax.scatter(PCA_df.loc[idxs, f'PC{chosen_PCs[0]}'],
                   PCA_df.loc[idxs, f'PC{chosen_PCs[1]}'],
                   c = color,s = 50, alpha=0.45)

    # add legend
    ax.legend(targets)
    ax.grid()
    return ax, targets

def print_top(df, attribute: str, n: int):
    print("-"*80)
    print(f"Top {n} ({attribute}) QA-pairs:")

    for row in df.sort_values(by=attribute, ascending=False)[:n].iterrows():
        subframe = pd.DataFrame(row[1:])

        print("-"*80)
        print(f"Question:\n {subframe.question_str.iloc[0]}\n")
        print(f"Answer:\n {Counter(subframe.answers.iloc[0])}\n")
        print(f"{attribute}: {subframe[attribute].iloc[0]}")

def print_bottom(df, attribute: str, n: int):
    print("-"*80)
    print(f"Bottom {n} ({attribute}) QA-pairs:")

    for row in df.sort_values(by=attribute, ascending=True)[:n].iterrows():
        subframe = pd.DataFrame(row[1:])

        print("-"*80)
        print(f"Question:\n {subframe.question_str.iloc[0]}\n")
        print(f"Answer:\n {Counter(subframe.answers.iloc[0])}\n")
        print(f"{attribute}: {subframe[attribute].iloc[0]}")


# define plotting function
def LogAndLinearHist(data, xlabel='', figsize=((8,3)), dpi=100):
    min_val, max_val = (min(data), max(data))
    
    # compute bins
    log_bins = np.logspace(min_val if min_val == 0 else np.log10(min_val), np.log10(max_val), 30)
    lin_bins = np.linspace(min_val, max_val, 30)

    # create histogram values
    hist_log, edges_log = np.histogram(data.values, log_bins, density=True)
    hist_lin, edges_lin = np.histogram(data.values, lin_bins)

    # determine x-values
    log_x = (edges_log[1:] + edges_log[:-1]) / 2.
    lin_x = (edges_lin[1:] + edges_lin[:-1]) / 2.

    xx, yy = zip(*[(i,j) for (i,j) in zip(log_x, hist_log) if j > 0])
    
    # plot figure
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=dpi)

    # linear scale plot
    ax[0].plot(lin_x, hist_lin, marker='.', alpha=0.5)
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel('count')
    ax[0].set_title('linear scale')
    #ax[0].legend()

    # log-log scale plot
    ax[1].plot(xx, yy, marker='.', alpha=0.5)
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel('probability density')
    ax[1].set_title('log-log scale')
    #ax[1].legend()

    # show figure
    plt.tight_layout()
    plt.show()

    
def investigate_this(anonymous_function, data_attribute: pd.Series, what=False):
    counter = Counter(data_attribute.apply(anonymous_function))
    attr, freqs = list(zip(*sorted(counter.items(), key=lambda x: x[1], reverse=True)))
    if type(attr[0]) == int:
        datalist = pd.Series(np.concatenate([[attr[i]]*freq for (i, freq) in enumerate(freqs)]))
    else:
        datalist = pd.Series(np.concatenate([[i]*freq for (i, freq) in enumerate(freqs)]))
    
    if what != False:
        print(f"Number of {what}: {attr.__len__()}")
    
        #from utils import LogAndLinearHist
        LogAndLinearHist(datalist, xlabel=what)
    
    # ccompute percentages
    percentage = pd.DataFrame({'attribute':attr, 'percentage':np.array(freqs)/sum(freqs)*100})
    datalist = pd.Series(np.concatenate([[attr[i]]*freq for (i, freq) in enumerate(freqs)]))
    
    return datalist, percentage
    

    
def createSunburstVariables(dataset, N, shuffle=True, seed=42):
    # define accumulation operator for strings
    def accum_operator(x1, x2, join_char=" "):
        return x1 + join_char + x2
    
    # initialize lists
    parents, labels, ids = [''], [''], ['CLS']
    occurrence_labels, occurrence_dict = [], {}

    
    # loop through token lists from questions
    for token_list in tqdm(dataset.sample(frac=1, random_state=seed)[:N].question_tokens):
        # create lists of accumulated strings
        accum = list(accumulate(['CLS']+token_list, func=accum_operator))
        accum_labels = list(accumulate(token_list, func=accum_operator))

        # define candidate tokens
        parent_candidates = accum[:-1]
        id_candidates = accum[1:]
        label_candidates = token_list

        # loop through candidate ids
        for i, id_candidate in enumerate(id_candidates):

            # count occurrences of this accumulated label
            try:
                occurrence_dict[accum_labels[i]] += 1
            except KeyError:
                occurrence_dict[accum_labels[i]] = 1

            # check if we have seen this id before (if so, it will cause trouble if we include it!)
            if id_candidate not in set(ids):
                parents.append(parent_candidates[i])
                ids.append(id_candidates[i])
                labels.append(token_list[i])
                occurrence_labels.append(accum_labels[i])

    # count occurrences of labels for list       
    occurrences = [int(occurrence_dict[id_name]) for id_name in occurrence_labels]
    occurrences.insert(0, N)
    
    df = pd.DataFrame({'ids':ids, 'labels':labels, 'parents':parents, 'occurrences':occurrences})
    df.occurrences.astype(int)

    return df

def plotSunburst(df, N, visualization_depth, width=500, height=500, use_dash=True):
    fig = go.Figure(
            go.Sunburst(
                ids=df.ids,
                labels=df.ids.apply(lambda seq: seq.strip("CLS")),
                parents=df.parents,
                values=df.occurrences,
                text=df.occurrences / N * 100,
                branchvalues="total",
                hovertemplate='<b>%{label} </b> <br> Percentage: %{text:.2f}%',
                domain=dict(column=1),
                maxdepth=visualization_depth,
                insidetextorientation='radial',
            ))

    
    #fig.update_traces(hovertemplate=hovertemplate)
    fig.update_layout(autosize=False, 
                      width=width, height=height, 
                      margin=dict(t=10, l=10, r=10, b=10), 
                      uniformtext=dict(minsize=10, mode='hide'))

    if use_dash:
        app = dash.Dash()
        app.layout = html.Div([dcc.Graph(figure=fig)])
        app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter
    else:
        fig.show()