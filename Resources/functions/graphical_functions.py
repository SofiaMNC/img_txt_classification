'''
    This module defines a set of graphical
    functions for project #6.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix

import helper_functions as hf

#------------------------------------------

def plot_percentage_missing_values_for(data, long, larg):
    '''
        Plots the proportions of filled / missing values for each unique value
        in column as a horizontal bar chart.

        Parameters
        ----------------
        data : pandas dataframe with:
                - a column column
                - a column "Percent Filled"
                - a column "Percent Missing"
                - a column "Total"

       long : int
            The length of the figure for the plot

        larg : int
               The width of the figure for the plot

        Returns
        ---------------
        -
    '''

    data_to_plot = hf.get_missing_values_percent_per(data)\
                     .sort_values("Percent Filled").reset_index()

    # Constants for the plot
    TITLE_SIZE = 60
    TITLE_PAD = 100
    TICK_SIZE = 50
    TICK_PAD = 20
    LABEL_SIZE = 50
    LABEL_PAD = 50
    LEGEND_SIZE = 50

    sns.set(style="whitegrid")

    #sns.set_palette(sns.dark_palette("purple", reverse=True))

    # Initialize the matplotlib figure
    _, axis = plt.subplots(figsize=(long, larg))

    plt.title("PROPORTIONS DE VALEURS RENSEIGNÉES / NON-RENSEIGNÉES PAR COLONNE",
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Plot the Total values
    handle_plot_1 = sns.barplot(x="Total", y="index",
                                data=data_to_plot,
                                label="non renseignées",
                                color="thistle", alpha=0.3)

    handle_plot_1.set_xticklabels(handle_plot_1.get_xticks(),
                                  size=TICK_SIZE)
    _, ylabels = plt.yticks()
    handle_plot_1.set_yticklabels(ylabels, size=TICK_SIZE)


    # Plot the Percent Filled values
    handle_plot_2 = sns.barplot(x="Percent Filled",
                                y="index",
                                data=data_to_plot,
                                label="renseignées",
                                color="darkviolet")

    handle_plot_2.set_xticklabels(handle_plot_2.get_xticks(),
                                  size=TICK_SIZE)
    handle_plot_2.set_yticklabels(ylabels, size=TICK_SIZE)


    # Add a legend and informative axis label
    axis.legend(bbox_to_anchor=(1.04, 0), loc="lower left",
                borderaxespad=0, ncol=1, frameon=True, fontsize=LEGEND_SIZE)

    axis.set(ylabel="Colonnes", xlabel="Pourcentage de valeurs (%)")

    x_label = axis.get_xlabel()
    axis.set_xlabel(x_label, fontsize=LABEL_SIZE,
                    labelpad=LABEL_PAD, fontweight="bold")

    y_label = axis.get_ylabel()
    axis.set_ylabel(y_label, fontsize=LABEL_SIZE,
                    labelpad=LABEL_PAD, fontweight="bold")

    axis.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,
                                                               pos: '{:2d}'.format(int(x)) + '%'))
    axis.tick_params(axis='both', which='major', pad=TICK_PAD)

    sns.despine(left=True, bottom=True)

    # Display the figure
    plt.show()

#------------------------------------------

def plot_word_number_proportions(data, long, larg):
    '''
        Plots the repartition of the words by text
        feature in data

        Parameters
        ----------------
        data : pandas dataframe
               The data to plot containing the cumsum of
               the number of words per category per
               state

        long : int
               The length of the figure for the plot

        larg : int
               The width of the figure for the plot

        Returns
        ---------------
        -
    '''

    # Constants for the plot
    TITLE_SIZE = 60
    TITLE_PAD = 100
    TICK_SIZE = 50
    TICK_PAD = 30
    LABEL_SIZE = 50
    LABEL_PAD = 50
    LEGEND_SIZE = 40

    # Reset index to access the category as a column
    data_to_plot = data.reset_index()

    sns.set(style="whitegrid")
    palette = sns.husl_palette(len(data.columns))

    # Initialize the matplotlib figure
    _, axis = plt.subplots(figsize=(long, larg))

    plt.title("RÉPARTITION DES MOTS PAR FEATURE TEXTE SUIVANT LA CATÉGORIE PRODUIT",
              fontweight="bold", fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Get the list of topics from the columns of data
    column_list = list(data.columns)

    # Create a barplot with a distinct color for each topic
    for idx, column in enumerate(reversed(column_list)):
        color = palette[idx]
        plot_handle = sns.barplot(x=column, y="category", data=data_to_plot,
                                  label=str(column), orient="h", color=color)

        plot_handle.set_xticklabels(plot_handle.get_xticks(), size=TICK_SIZE)
        _, ylabels = plt.yticks()
        plot_handle.set_yticklabels(ylabels, size=TICK_SIZE)

    # Add a legend and informative axis label
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles[::-1], labels[::-1],
                bbox_to_anchor=(0, -0.4, 1, 0.2),
                loc="lower left", mode="expand",
                borderaxespad=0, ncol=4, frameon=True,
                fontsize=LEGEND_SIZE)
    
    

    axis.set(ylabel="category", xlabel="% de mots par catégorie")

    x_label = axis.get_xlabel()
    axis.set_xlabel(x_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    y_label = axis.get_ylabel()
    axis.set_ylabel(y_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    axis.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:2d}'.format(int(x))))

    axis.tick_params(axis='both', which='major', pad=TICK_PAD)

    sns.despine(left=True, bottom=True)

    # Display the figure
    plt.show()

#------------------------------------------

def display_scree_plot(pca):
    '''
        Plots the scree plot for the given pca
        components.

        ----------------
        - pca : A PCA object
                The result of a PCA decomposition

        Returns
        ---------------
        _
    '''

    # Constants for the plot
    TITLE_SIZE = 30
    LABEL_SIZE = 20
    LABEL_PAD = 30

    plt.subplots(figsize=(10, 10))

    scree = pca.explained_variance_ratio_ * 100

    plt.bar(np.arange(len(scree))+1, scree)

    plt.plot(np.arange(len(scree))+1,
             scree.cumsum(), c="red", marker='o')

    plt.xlabel("Rang de l'axe d'inertie",
               fontsize=LABEL_SIZE,
               labelpad=LABEL_PAD)

    plt.ylabel("% d'inertie",
               fontsize=LABEL_SIZE,
               labelpad=LABEL_PAD)

    plt.title("Eboulis des valeurs propres",
              fontsize=TITLE_SIZE)

    plt.show(block=False)

#------------------------------------------

def plot_freq_dist(data_df, title, long, larg):
    '''
        Displays a bar chart showing the frequency of the modalities
        for each column of data.

        Parameters
        ----------------
        data  : dataframe
                Working data containing exclusively qualitative data
               
        title : string
                The title to give the plot

        long  : int
                The length of the figure for the plot

        larg  : int
                The width of the figure for the plot

        Returns
        ---------------
        -
    '''

    TITLE_SIZE = 40
    TITLE_PAD = 80
    TICK_SIZE = 12
    LABEL_SIZE = 30
    LABEL_PAD = 20

    # Initialize the matplotlib figure
    _, axis = plt.subplots(figsize=(long, larg))

    plt.title(title,
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Plot the Total values
    data_to_plot = data_df.reset_index().rename(columns={"index":"words"})
    handle_plot_1 = sns.barplot(x="words", y="freq", data=data_to_plot,
                                label="non renseignées", color="darkviolet", alpha=1)

    _, xlabels = plt.xticks()
    _ = handle_plot_1.set_xticklabels(xlabels, size=TICK_SIZE, rotation=45)
    
    x_label = axis.get_xlabel()
    axis.set_xlabel(x_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    y_label = axis.get_ylabel()
    axis.set_ylabel(y_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

#------------------------------------------

def plot_pix_histogram(img, is_img_name=False, long=15, larg=10):
    '''
        Plots an histogram for the given image

        Parameters
        ----------------
        img         : PIL image or string
        
        is_img_name : Bool
                      Whether img is the path+name of the img
                      or the img itself

        long        : int
                      The length of the figure for the plot

        larg        : int
                      The width of the figure for the plot

        Returns
        ---------------
        -
    '''

    # Constants for the plot
    TITLE_SIZE = 30
    TITLE_PAD = 1
    TICK_SIZE = 20
    LABEL_SIZE = 20
    LABEL_PAD = 30
    LEGEND_SIZE = 15

    sns.set_palette(sns.dark_palette("purple", reverse=True))
    sns.set(style="white")

    plt.rcParams["font.weight"] = "bold"

    plt.rc('font', size=LABEL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LABEL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LABEL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)

    fig, axis = plt.subplots(figsize=(15, 10))

    fig.suptitle("Histogramme image",
                 fontweight="bold",
                 fontsize=TITLE_SIZE, y=TITLE_PAD)
                 
    if is_img_name:
        img_histo_to_plot = Image.open(img)
    else:
        img_histo_to_plot = img
        
    _ = sns.lineplot(x="Pixel value",
                     y="Freq",
                     data=pd.DataFrame(data=img_histo_to_plot.convert("L").histogram())\
                            .reset_index()\
                            .rename(columns={0:"Freq","index":"Pixel value"}))

    x_label = axis.get_xlabel()
    axis.set_xlabel(x_label.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    y_label = axis.get_ylabel()
    axis.set_ylabel(y_label.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

#------------------------------------------

def plot_qualitative_dist(data, nb_rows, nb_cols, long, larg, title):
    '''
        Displays a bar chart showing the frequency of the modalities
        for each column of data.

        Parameters
        ----------------
        data : dataframe
               Working data containing exclusively qualitative data

        long : int
               The length of the figure for the plot

        larg : int
               The width of the figure for the plot

        Returns
        ---------------
        -
    '''

    # Contants for the plot
    TITLE_SIZE = 130
    TITLE_PAD = 1.05
    TICK_SIZE = 50
    LABEL_SIZE = 80
    LABEL_PAD = 30

    fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(long, larg))

    fig.suptitle("DISTRIBUTION DES VALEURS QUALITATIVES"+" "+title,
                 fontweight="bold", fontsize=TITLE_SIZE, y=TITLE_PAD)

    row = column = 0
    
    for ind_qual in data.columns.tolist():
        
        data_to_plot = data.sort_values(by=ind_qual).copy()
        
        if(nb_rows == 1 and nb_cols == 1):
            axis = axes
        elif(nb_rows == 1 or nb_cols == 1):
            if nb_rows == 1:
                axis = axes[column]
            else:
                axis = axes[row]
        else:
            axis = axes[row, column]

        plot_handle = sns.countplot(y=ind_qual,
                                    data=data_to_plot,
                                    color="darkviolet",
                                    ax=axis,
                                    order=data_to_plot[ind_qual].value_counts().index)

        plt.tight_layout()

        plt.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=1.4, hspace=0.2)

        plot_handle.set_xticklabels(plot_handle.get_xticks(), size=TICK_SIZE)
    
        yticks = [item.get_text().upper() for item in axis.get_yticklabels()]
        plot_handle.set_yticklabels(yticks, size=TICK_SIZE, weight="bold")
        
        x_label = axis.get_xlabel()
        axis.set_xlabel(x_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        y_label = axis.get_ylabel()
        axis.set_ylabel(y_label.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        axis.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x))))

        axis.xaxis.grid(True)

        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0

#------------------------------------------

def plot_correlation_circle(pcs, labels, long, larg):
    '''
        Plots 2 distplots horizontally in a single figure

        Parameters
        ----------------
        pcs     : PCA components
                  Components from a PCA

        labels  : list
                  The names to give the components

        long            : int
                          The length of the figure for the plot

        larg            : int
                          The width of the figure for the plot

        Returns
        ---------------
        -
    '''

    # Constants for the plot
    TITLE_SIZE = 30
    TITLE_PAD = 70

    plt.subplots(figsize=(long, larg))

    for i, (x_value, y_value) in enumerate(zip(pcs[0, :], pcs[1, :])):
        if(x_value > 0.2 or y_value > 0.2):
            plt.plot([0, x_value], [0, y_value], color='k')
            plt.text(x_value, y_value, labels[i], fontsize='14')

    plt.plot([-0.5, 0.5], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-0.5, 0.5], color='grey', ls='--')

    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])

    plt.title("Corrélations - Variables latentes",
              fontsize=TITLE_SIZE,
              pad=TITLE_PAD)

    plt.subplots(figsize=(long, larg))

    for i, (x_value, y_value) in enumerate(zip(pcs[2, :], pcs[3, :])):
        if(x_value > 0.2 or y_value > 0.2):
            plt.plot([0, x_value], [0, y_value], color='k')
            plt.text(x_value, y_value, labels[i], fontsize='14')

    plt.plot([-0.5, 0.5], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-0.5, 0.5], color='grey', ls='--')

    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])

    plt.title("Corrélations - Variables latentes",
              fontsize=TITLE_SIZE,
              pad=TITLE_PAD)

    plt.subplots(figsize=(long, larg))

    for i, (x_value, y_value) in enumerate(zip(pcs[4, :], pcs[5, :])):
        if(x_value > 0.2 or y_value > 0.2):
            plt.plot([0, x_value], [0, y_value], color='k')
            plt.text(x_value, y_value, labels[i], fontsize='14')

    plt.plot([-0.5, 0.5], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-0.5, 0.5], color='grey', ls='--')

    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])
    plt.title("Corrélations - Variables latentes",
              fontsize=TITLE_SIZE,
              pad=TITLE_PAD)

#------------------------------------------

def plot_bovw_histo(bovw_histo, index_img, title, long, larg):
    '''
        Plots the bag of visual words histogram
        for a given category

        Parameters
        ----------------
        bovw_histo : pandas dataframe
                     format : 1 row per image :
                     columns : indexes of the dictionary words
                              + image
                              + category
                              + subcategory
                              
        index_img   : int
                      index of image
        
        title       : string
                      The title of the plot
                              
        long        : int
                      The length of the figure for the plot

        larg        : int
                      The width of the figure for the plot

        Returns
        ---------------
        -
    '''

    data_to_plot = bovw_histo.iloc[:, :-3].T.reset_index()

    # Constants for the plot
    TITLE_SIZE = 100
    TITLE_PAD = 1.1
    TICK_SIZE = 40
    LABEL_SIZE = 80
    LABEL_PAD = 30

    fig, ax = plt.subplots(figsize=(long, larg))

    plt.title(title,
                fontsize=TITLE_SIZE,
                y=TITLE_PAD)
             
    plot_handle = sns.barplot(x="index",
                              y=index_img,
                              data=data_to_plot,
                              color="darkviolet")

    plot_handle.set_xticklabels(plot_handle.get_xticks(), size=TICK_SIZE)

    plot_handle.set_yticklabels(plot_handle.get_yticks(), size=TICK_SIZE)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x))))

    x_label = ax.get_xlabel()
    ax.set_xlabel(x_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    y_label = ""

    change_width(ax, .45)
    
#------------------------------------------

def change_width(ax, new_value):
    '''
        Change the width of the bars in a
        barplot and modifies the xtick
        positions accordingly.

        Parameters
        ----------------
        ax       : the plot's axis

        new_value: int
                    The ratio of the original
                    width to use as the new
                    width

        Returns
        ---------------
        -
    '''

    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

#------------------------------------------

def plot_cm(y_true, y_pred, subtitle, figsize=(10,10)):
    '''
        Plots a confusion matrix as a heatmap
        for the entry data

        Parameters
        ----------------
        y_true : list
                 actual values

        y_pred : list
                 predicted values
        
        subtitle : string
                   the second part of the title
                   the name of the data for which
                   the confusion matrix is plotted
                   
        figsize : tuple
                  size of the plot

        Returns
        ---------------
        -
    '''
    
    TITLE_SIZE = 20
    TITLE_PAD = 40
    LABEL_SIZE = 15
    LABEL_PAD = 50
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'ACTUAL'
    cm.columns.name = 'PREDICTED'
    
    title = "MATRICE DE CONFUSION\nDonnées : " + subtitle
    
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title,
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)
    
    sns.heatmap(cm, cmap= "Purples", annot=annot, fmt='', ax=ax)
    
    x_label = ax.get_xlabel()
    ax.set_xlabel(x_label, fontsize=LABEL_SIZE,
                 labelpad=LABEL_PAD, fontweight="bold")
    
    y_label = ax.get_ylabel()
    ax.set_ylabel(y_label, fontsize=LABEL_SIZE,
                 labelpad=LABEL_PAD, fontweight="bold")
