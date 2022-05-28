
import os
import torchvision
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

from pathlib import Path
from sklearn.manifold import TSNE

import pandas as pd

def plot_example(input,
                 saliency,
                 method,
                 category_id,
                 answer_vocab: list,
                 show_plot=False,
                 save_path=None,
                 analysis_type=None):
    """Plot an example.

    Args:
        input (:class:`torch.Tensor`): 4D tensor containing input images.
        saliency (:class:`torch.Tensor`): 4D tensor containing saliency maps.
        method (str): name of saliency method.
        category_id (int): ID of ImageNet category.
        show_plot (bool, optional): If True, show plot. Default: ``False``.
        save_path (str, optional): Path to save figure to. Default: ``None``.
    """
    from torchray.utils import imsc

    if isinstance(category_id, int):
        category_id = [category_id]

    batch_size = len(input)

    plt.clf()
    for i in range(batch_size):
        class_i = category_id[i % len(category_id)]

        plt.subplot(batch_size, 2, 1 + 2 * i)
        imsc(input[i])
        plt.title(f'{analysis_type}', fontsize=8)

        plt.subplot(batch_size, 2, 2 + 2 * i)
        imsc(saliency[i], interpolation='none')
        plt.title('{} for answer category {} (id: {})'.format(
            method, answer_vocab[class_i], class_i), fontsize=8)

    # Save figure if path is specified.
    if save_path:
        save_dir = os.path.dirname(os.path.abspath(save_path))
        # Create directory if necessary.
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ext = os.path.splitext(save_path)[1].strip('.')
        plt.savefig(save_path, format=ext, bbox_inches='tight', dpi=500)

    # Show plot if desired.
    if show_plot:
        plt.show()



def plot_stratified_results(stratified_object, barplot_dict, strat_type,
                            args,
                            model_name: str):
    
    # Compute t-SNE
    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(stratified_object.embeddings.T)
    
    # Restructure barplot-dictionary
    newdict = {(k1, k2):v2 for k1,v1 in barplot_dict.items() \
               for k2,v2 in barplot_dict[k1].items()}  
    df = pd.DataFrame([newdict[i] for i in sorted(newdict)],
                      index=pd.MultiIndex.from_tuples([i for i in sorted(newdict.keys())]))
    # CIs
    df['low'] = df['avg'] - df.CIs.apply(lambda x: x[0])
    df['high'] = df['avg'] + df.CIs.apply(lambda x: x[1])
     
    # Name formatting                   
    df = df.reset_index().rename(columns={'level_0': 'label', 'level_1': 'score'}).set_index(['label', 'score'])
    df = df.unstack(level='label')
    df = df.rename(index={'vqa_acc': 'VQA Acc.', 'numberbatch_score': 'Numberbatch', 'acc': 'Accuracy'})
    df = df.drop('Accuracy')
    
    # Scatter plot with colors based on the stratification_func
    cmap = cm.get_cmap('tab20')
    colors = [cmap(stratified_object.cat2idx[cat]) for cat in stratified_object.categories]

    fig, ax = plt.subplots(1, 2, figsize=(15,10), dpi=400,
                           gridspec_kw={'width_ratios': [2, 1]})
    for i,cat in enumerate(stratified_object.categories):
        indices = stratified_object.data['stratification_label'][stratified_object.data['stratification_label'] == cat].index.to_numpy()
        color = colors[i]
        ax[0].scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(color).reshape(1,4), label = cat ,alpha=0.5)
        
    #ax[0].legend(fontsize='large', markerscale=2)
    
    # Set title and legend
    ax[0].axis('off')
    ax[0].set_title(f"Qlarifais-model: {model_name}\
                    \nPredictions stratified by {stratified_object.by}",
                    fontsize=20,
                    loc='left')
    #plt.legend(loc='center', bbox_to_anchor=(0.95, 0.3, 0.5, 0.5))
    plt.tight_layout()
    
    # Plot bars
    df['avg'].plot(kind='barh', xerr=df[['low','high']].T.values,
                     width=0.88, color=colors,
                     #figsize=(10,8), 
                     ax=ax[1])
    #ax[1].set_title(f"Stratified by: {strat_type}", fontsize=15, loc='left')
    #ax[1].legend(loc='center', fontsize=20,
    #             bbox_to_anchor=(1.05, 0.3, 0.5, 0.5))
    ax[1].legend(loc='upper center', fontsize=18,
             bbox_to_anchor=(-0.8, -0.05),fancybox=False, 
             shadow=False, ncol=2)
    ax[1].set_ylabel('')
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=16)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    #plt.tight_layout()
    fig.show()
    
    os.makedirs(Path(args.save_path) / f'figures', exist_ok=True)
    plt.savefig(Path(args.save_path) / f'figures/{strat_type}.png') 
    