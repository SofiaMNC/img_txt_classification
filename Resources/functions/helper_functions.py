'''
    This module defines a set of calculations
    functions for project 5.
'''

import time
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PIL import Image, ImageOps, ImageFilter
import cv2 as cv
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn import (preprocessing,
                     manifold,
                     decomposition)

#------------------------------------------

def get_missing_values_percent_per(data):
    '''
        Calculates the mean percentage of missing values
        in a given pandas dataframe per unique value
        of a given column

        Parameters
        ----------------
        data                : pandas dataframe
                              The dataframe to be analyzed

        Returns
        ---------------
        missing_percent_df  : A pandas dataframe containing:
                                - a column "column"
                                - a column "Percent Missing" containing the percentage of
                                  missing value for each value of column
    '''

    missing_percent_df = pd.DataFrame({'Percent Missing':data.isnull().sum()/len(data)*100})
    missing_percent_df['Percent Filled'] = 100 - missing_percent_df['Percent Missing']
    missing_percent_df['Total'] = 100

    return missing_percent_df


#------------------------------------------

def describe_dataset(source_files):
    '''
        Outputs a presentation pandas dataframe for the dataset.

        Parameters
        ----------------
        sourceFiles     : dict with :
                            - keys : the names of the files
                            - values : a list containing two values :
                                - the dataframe for the data
                                - a brief description of the file

        Returns
        ---------------
        presentation_df : pandas dataframe :
                            - a column "Nom du fichier" : the name of the file
                            - a column "Nb de lignes"   : the number of rows per file
                            - a column "Nb de colonnes" : the number of columns per file
                            - a column "Description"    : a brief description of the file
    '''

    print("Les données se décomposent en {} fichier(s): \n".format(len(source_files)))

    filenames = []
    files_nb_lines = []
    files_nb_columns = []
    files_descriptions = []

    for filename, file_data in source_files.items():
        filenames.append(filename)
        files_nb_lines.append(len(file_data[0]))
        files_nb_columns.append(len(file_data[0].columns))
        files_descriptions.append(file_data[1])

    # Create a dataframe for presentation purposes
    presentation_df = pd.DataFrame({'Nom du fichier':filenames,
                                    'Nb de lignes':files_nb_lines,
                                    'Nb de colonnes':files_nb_columns,
                                    'Description': files_descriptions})

    presentation_df.index += 1

    return presentation_df

#------------------------------------------

def assign_random_clusters(data, n_clusters):
    '''
        Assigns a random cluster out of n_clusters to each observation
        in data.

        Parameters
        ----------------
        data             : pandas dataframe
                           Contains the observations

        - n_clusters    : int
                          Number of clusters to choose from

        Returns
        ---------------
        _   : numpy array
              Contains the assigned clusters for the observations in data
    '''

    return np.random.randint(n_clusters, size=len(data))

#------------------------------------------

def fit_plot(algorithms, data, long, larg, title):
    '''
        For each given algorithm :
        - fit them to the data on 3 iterations
        - Calculate the mean silhouette and adjusted rand scores
        - Gets the calculation time

        The function then plots the identified clusters for each algorithm.

        Parameters
        ----------------
        algorithms : dictionary with
                        - names and type of input as keys
                        - instantiated algorithms as values

        - data     : pandas dataframe
                     Contains the data to fit the algos on

        - long     : int
                     length of the plot figure

        - larg     : int
                     width of the plot figure

        - title    : string
                     title of the plot figure

        Returns
        ---------------
        scores_time : pandas dataframe
                      Contains the mean silhouette coefficient,
                      the adjusted Rnad score, the number of clusters,
                      the calculation time for each algorithm in algorithms
    '''

    scores_time = pd.DataFrame(columns=["Algorithme", "iter",
                                        "silhouette", "Rand",
                                        "Nb Clusters", "Time"])

    # Constants for the plot
    TITLE_SIZE = 60
    TITLE_PAD = 1.05
    SUBTITLE_SIZE = 40
    TICK_SIZE = 25
    LABEL_SIZE = 30
    LABEL_PAD = 30
    LEGEND_SIZE = 40

    nb_rows = int(len(algorithms)/2) if int(len(algorithms)/2) > 2 else 1
    nb_cols = 2

    fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(larg, long))
    fig.suptitle(title, fontweight="bold", fontsize=TITLE_SIZE, y=TITLE_PAD)

    row = column = 0
    ITER = 3 # constant

    for algoname, algo in algorithms.items():

        cluster_labels = {}

        for i in range(ITER):
            if algoname == "Dummy":
                start_time = time.time()
                cluster_labels[i] = assign_random_clusters(data, algo)
                elapsed_time = time.time() - start_time
            else:
                start_time = time.time()
                algo.fit(data)
                elapsed_time = time.time() - start_time
                cluster_labels[i] = algo.labels_

        for i in range(ITER):
            j = i+1

            if i == 2:
                j = 0

            scores_time.loc[len(scores_time)] = [algoname, i,
                                                 silhouette_score(data,
                                                                  cluster_labels[i],
                                                                  metric="euclidean"),
                                                 adjusted_rand_score(cluster_labels[i],
                                                                     cluster_labels[j]),
                                                 len(set(cluster_labels[i])),
                                                 elapsed_time]

        # plot
        if nb_rows == 1:
            axis = axes[column]
        else:
            axis = axes[row, column]

        data_to_plot = data.copy()
        data_to_plot["cluster_labels"] = cluster_labels[ITER-1]
        plot_handle = sns.scatterplot(x="tsne-one-tf", y="tsne-two-tf",
                                      data=data_to_plot, hue="cluster_labels",
                                      palette=sns\
                                      .color_palette("hls",
                                                     data_to_plot["cluster_labels"].nunique()),
                                      legend="full", alpha=0.3, ax=axis, s=500)

        plt.tight_layout()

        plt.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=0.3, hspace=0.4)

        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.get_xaxis().tick_bottom()
        axis.get_yaxis().tick_left()
        axis.spines['left'].set_position(('outward', 10))
        axis.spines['bottom'].set_position(('outward', 10))

        axis.set_xlabel('tsne-pca-one', fontsize=LABEL_SIZE, labelpad=LABEL_PAD)
        axis.set_ylabel('tsne-pca-two', fontsize=LABEL_SIZE, labelpad=LABEL_PAD)

        plot_handle.set_xticklabels(plot_handle.get_xticks(), size=TICK_SIZE)

        plot_handle.set_yticklabels(plot_handle.get_yticks(), size=TICK_SIZE)
        axis.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x)))

        extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                              edgecolor='none', linewidth=0)

        scores = (r'$Silh={:.2f}$' + '\n' + r'$Rand={:.2f}$')\
                 .format(scores_time[scores_time["Algorithme"] == algoname]["silhouette"].mean(),
                         scores_time[scores_time["Algorithme"] == algoname]["Rand"].mean())

        axis.legend([extra], [scores], loc='upper left', fontsize=LEGEND_SIZE)
        title = algoname + '\n Évaluation en {:.2f} secondes'.format(elapsed_time)
        axis.set_title(title, fontsize=SUBTITLE_SIZE, fontweight="bold")

        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0

    return scores_time

#------------------------------------------

def lemmatized_list(word_list):
    '''
        Lemmatizes the words in the input list

        Parameters
        ----------------
        word_list : list
                    The words to be lemmatized

        Returns
        ---------------
        lemmatized_list : list
                          The lemmatized words
                
    '''
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_list = []

    for word in word_list:
        lemmatized_list.append(lemmatizer.lemmatize(word))

    return lemmatized_list

#------------------------------------------

def get_most_freq(data_series, nb):
    '''
        Counts the occurrences of each words in data_series
        and returns the nb most frequent with their associated
        count.
        
        Parameters
        ----------------
        data_series : pandas series
                      The corpus of documents

        - nb        : int
                      The number of most frequent words to
                      return

        Returns
        ---------------
        df   : pandas dataframe
               The nb most frequent words with their associated
               count.
    '''
    
    all_words = []

    for word_list in data_series:
        all_words += word_list
        
    freq_dict = nltk.FreqDist(all_words)

    df = pd.DataFrame.from_dict(freq_dict, orient='index').rename(columns={0:"freq"})

    return df.sort_values(by="freq", ascending=False).head(50)

#------------------------------------------

def get_nb_words_df(data, groupby_feature, feature):
    '''
        Counts the words in feature in data grouped by
        groupby_feature.

        Parameters
        ----------------
        data              : pandas dataframe
                            Contains the text

        - groupby_feature : string
                            The name of the feature in data
                            by which to group by
        
        - feature         : string
                            The name of the feature in data in
                            which to count
    
        Returns
        ---------------
        nb_words_df   : pandas dataframe
                        Contains the words and their count in feature, per
                        groupby_feature.
    '''
    
    # Total number of words

    all_words_prod_name = []

    for word_list in data[feature]:
        all_words_prod_name += word_list

    total_words_prod_name = len(all_words_prod_name)

    # Number of words per category

    cat_words_prod_name = {}

    for cat1, data_df in data.groupby(groupby_feature):
        cat_words = []

        for word_list in data_df[feature]:
            cat_words += word_list

        cat_words_prod_name[cat1] = cat_words
        

    # Setting up return DataFrame

    nb_words_df = pd.DataFrame()

    nb_words_df[groupby_feature] = data[groupby_feature].unique()
    nb_words_df["Nb_Words"] = nb_words_df[groupby_feature].apply(lambda x: len(cat_words_prod_name[x]))
    nb_words_df["Difference"] = nb_words_df["Nb_Words"].apply(lambda x: total_words_prod_name - x)
    nb_words_df["Total_Words"] = total_words_prod_name

    return nb_words_df

#------------------------------------------

def print_top_words(model, feature_names, n_top_words):
    '''
        Prints the n_top_words found by a text analysis
        model (LDA or NMF)

        Parameters
        ----------------
        - model         : a fitted LDA or NMF model

        - feature_names : list
                          The ordered name of the features
                          used by the data the model
                          was fitted on

        Returns
        ---------------
        _
    '''
    
    for topic_idx, topic in enumerate(model.components_):
        print("Catégorie #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

    print()
    
#------------------------------------------

def calculate_exposition_coeff(img_serie, path=None):
    '''
        Calculates the exposition coefficient, that gives the % of
        imbalanced pixels. A negative sign denotes an imbalance in
        favor of dark pixels, while a positive indicates an imbalance
        in favor or light pixels.

        Parameters
        ----------------
        - img_serie : pandas series
                      The pictures or the pictures' names.

        - path      : string
                      The path to load the picture in case
                      img_serie contains the pictures' name.

        Returns
        ---------------
        exposition_coeff_df : pandas dataframe
                              The images and their exposition coefficient
    '''
    
    exposition_coeff_df = pd.DataFrame(columns=["exposition_coeff"])

    for img in img_serie:
        
        if path == None:
            subj_img = img
        else:
            subj_img = Image.open(path+img)
            
        h = subj_img.convert("L").histogram()
        nb_pix_under_128 = np.sum(h[:128])
        nb_pix_128_and_over = np.sum(h[128:])

        exposition_coeff = ((nb_pix_128_and_over-nb_pix_under_128)\
                            /\
                           (nb_pix_under_128+nb_pix_128_and_over))*100

        exposition_coeff_df.loc[len(exposition_coeff_df)] = [exposition_coeff]
        
    return exposition_coeff_df

#------------------------------------------

def calculate_variation_coeff(img_serie, path=None):
    '''
        Calculates the variation coefficient for the
        distribution of the pixels in the image.

        Parameters
        ----------------
        - img_serie : pandas series
                      The pictures or the pictures' names.

        - path      : string
                      The path to load the picture in case
                      img_serie contains the pictures' name.

        Returns
        ---------------
        variation_coeff_df : pandas dataframe
                             The images and their variation coefficient
    '''
    
    
    variation_coeff_df = pd.DataFrame(columns=["variation_coeff"])
    
    for img in img_serie:
    
        if path == None:
            subj_img = img
        else:
            subj_img = Image.open(path+img)
            
        h = subj_img.convert("L").histogram()
        
        variation_coeff = np.std(h)/np.mean(h)
        
        variation_coeff_df.loc[len(variation_coeff_df)] = [variation_coeff]
        
    return variation_coeff_df

#------------------------------------------

def get_img_quality_coeffs(img_serie, path=None):
    '''
        Get the exposition and variation coefficients
        for the input images.

        Parameters
        ----------------
        - img_serie : pandas series
                      The pictures or the pictures' names.

        - path      : string
                      The path to load the picture in case
                      img_serie contains the pictures' name.

        Returns
        ---------------
        _ : pandas dataframe
            The images and their exposition and variation
            coefficients
    '''
    
    return pd.concat([calculate_exposition_coeff(img_serie, path),
                      calculate_variation_coeff(img_serie, path)],
                     axis=1)

#------------------------------------------

def to_value_list(list_of_strings):
    '''
        Create list of values from a list of source strings

        Parameters
        ----------------
        - list_of_strings : list of strings
                            The strings are formatted according
                            to the following :
                            ['{"key"=>"Brand", "value"=>"Elegance"}',
                             '{"key"=>"Designed For", "value"=>"Door"}]}']

        Returns
        ---------------
        _ : list of strings
            The values contained in the source strings
    '''
    
    value_list = []
    
    for str_key_value in list_of_strings:
        
        word = '"value":"'
        
        end_index = str_key_value.find(word) + len(word)
        
        remaining_word = str_key_value[end_index:-1]

        value_list.append(remaining_word)
    
    return value_list
    
#------------------------------------------

def get_value_list(pseudo_dict_string_serie):
    '''
        Create list of values from a source
        string

        Parameters
        ----------------
        pseudo_dict_string_serie : pandas serie
                                   Contains list as as string formatted
                                   according to the following :
                                   ['{"product_specification"=>[{"key"=>"Brand",
                                   "value"=>"Elegance"}, {"key"=>"Designed For",
                                   "value"=>"Door"}]}']

        Returns
        ---------------
        pseudo_dict_string_serie   : list
                                     Contains the values in the pseudo
                                     dictionary
    '''
    
    # Replacing "=>" with ":" and splitting around "["
    pseudo_dict_string_serie = pseudo_dict_string_serie\
                                         .replace({"=>":":"}, regex=True)\
                                         .str.split("[")


    # Keeping only the parts after the "product specifications : " string
    pseudo_dict_string_serie = pseudo_dict_string_serie\
                                         .apply(lambda x:x[1] if len(x)>1 else x[0])


    # Splitting to get a list of all key-value pairs
    pseudo_dict_string_serie = pseudo_dict_string_serie.str\
                                        .split("}, ")


    # Transforming the string to a list of values
    pseudo_dict_string_serie = pseudo_dict_string_serie\
                                         .apply(lambda x: to_value_list(x))

    # Set to type string
    pseudo_dict_string_serie = pseudo_dict_string_serie.astype("str")
    
    return pseudo_dict_string_serie

#------------------------------------------

def get_categories(category_tree_serie):
    '''
        Create list of values from a source
        string

        Parameters
        ----------------
        category_tree_serie : pandas series
                              Contains a list as as string formatted
                              according to the following :
                               ['["Home Furnishing >> Curtains & Accessories
                               >> Curtains
                               >> Elegance Polyester Multicolor Abstract Eyelet Do..."]']

        Returns
        ---------------
        categories   : pandas dataframe
                       A 2-column dataframe containing the first
                       and second level categories
    '''
    
    categories = pd.DataFrame(columns=["category",
                                       "subcategory"])

    # Populating category 1

    categories["category"] = category_tree_serie.astype("str")\
                             .apply(lambda x: list(x.split(">>"))[0])


    # Populating category 2

    categories["subcategory"] = category_tree_serie.astype("str")\
                                .apply(lambda x: list(x.split(">>"))[1]\
                                                     if len(list(x.split(">>"))) > 1\
                                                     else np.NaN)

    return categories

#------------------------------------------

def getKeypointsDescriptors(image, nb_keypoints):
    '''
        Get nb_keypoints keypoints and their associated
        descriptors from image.

        Parameters
        ----------------
        - image         : a picture

        - nb_keypoints  : int
                          Number of keypoints to find in
                          image

        Returns
        ---------------
        _   : tuple of arrays
              first item contains the keypoints
              second item contains the descriptors
    '''
    
    # Read img from dataframe
    opencv_img = np.array(image)[:, :, ::-1].copy()

    #Grayscale
    img = cv.cvtColor(opencv_img, cv.COLOR_BGR2GRAY)

    # Sift
    sift = cv.xfeatures2d.SIFT_create(nb_keypoints)
    
    return sift.detectAndCompute(img,None)

#------------------------------------------

def assembleKeypointsDescriptors(df, image_feature_name, nb_keypoints):
    '''
        Create 2 pandas dataframe containing the keypoints, respective
        the descriptors, for the list of images contained in the column
        image_feature_name in the df dataframe.

        Parameters
        ----------------
        - df                 : pandas dataframe
                               Contains the images, their category and
                               subcategory

        - image_feature_name : string
                               Name of the column in df containing
                               the images
        
        - nb_keypoints     : int
                             Number of keypoints to find in each image
        

        Returns
        ---------------
        _   : tuple of pandas dataframe
              first dataframe contains the keywords
              second dataframe contains the descriptors,
                                        the image,
                                        the category of the image,
                                        the subcategory of the image
    '''
    
    keypoints_df = pd.DataFrame()
    descriptors_df = pd.DataFrame()

    for img in df[image_feature_name]:
        
        keypoints_1, descriptors_1 = getKeypointsDescriptors(img, nb_keypoints)
                        
        # Saving keypoints and descriptors
        keypts_df = pd.DataFrame(data=keypoints_1)
        # This prevents image doubles
        keypts_df["image"] = df[df[image_feature_name]==img]["image"].iloc[0]
        keypts_df[image_feature_name] = df[df[image_feature_name]==img][image_feature_name].iloc[0]
        keypts_df["category"] = df[df[image_feature_name]==img]["category"].iloc[0]
        keypts_df["subcategory"] = df[df[image_feature_name]==img]["subcategory"].iloc[0]
        keypts_df = keypts_df.loc[:nb_keypoints-1,:]
        keypoints_df = pd.concat([keypoints_df, keypts_df])
        
                
        descs_df = pd.DataFrame(data=descriptors_1)
        # This prevents image doubles
        descs_df["image"] = df[df[image_feature_name]==img]["image"].iloc[0]
        descs_df[image_feature_name] = df[df[image_feature_name]==img][image_feature_name].iloc[0]
        descs_df["category"] = df[df[image_feature_name]==img]["category"].iloc[0]
        descs_df["subcategory"] = df[df[image_feature_name]==img]["subcategory"].iloc[0]
        descs_df = descs_df.loc[:nb_keypoints-1, :]
        descriptors_df = pd.concat([descriptors_df, descs_df])
        

    keypoints_df = keypoints_df.reset_index(drop=True)
    descriptors_df = descriptors_df.reset_index(drop=True)
    
    return (keypoints_df, descriptors_df)

#------------------------------------------

def getImgsHisto(descriptors, dictionary):
    '''
        Get the bag of visual words histogram for the
        input descriptors and dictionary

        Parameters
        ----------------
        - descriptors : pandas dataframe
                        The descriptors for the images
                          

        - dictionary  : array
                        The visual word dictionary

        Returns
        ---------------
        - histo   : pandas dataframe
                    The bag of visual words histogram (1 row)
                    for each image ( len(dictionary) columns +
                                 colum "image" +
                                 column "category" +
                                 column "subcategory"
    '''
    
    names = [x for x in range(len(dictionary))]
    names.append("category")
    names.append("subcategory")
    names.append("image")
    histo = pd.DataFrame(columns=names)
    
    for name_img, descs_df in descriptors.groupby("image"):
            
        # Determine for each descriptor the word in the dictionay
        # it is closest to.
        
        closest, _ = pairwise_distances_argmin_min(descs_df.iloc[:,:-4],
                                                   dictionary)
        
        hist = np.zeros(len(dictionary))

        for ind in closest:
            hist[ind] += 1
        
        id_info = [descs_df["category"].unique()[0],
                   descs_df["subcategory"].unique()[0],
                   name_img]
        
        hist = np.append(hist,id_info)
                
        histo.loc[len(histo)] = hist
            
    return histo
    
#------------------------------------------

def unit_tests():
    '''
        Runs unit test on two functions :
        - get_value_list
        - get_categories
        
        Prints the result of the tests.

        Parameters
        ----------------
        -

        Returns
        ---------------
        _
        
    '''
    
    test_df = pd.DataFrame()
    test_df["product_specifications"] = \
    ['{"product_specification"=>[{"key"=>"Brand", "value"=>"Elegance"}, {"key"=>"Designed For", "value"=>"Door"}]}']
    test_df["product_category_tree"] = \
    ['["Home Furnishing >> Curtains & Accessories >> Curtains >> Elegance Polyester Multicolor Abstract Eyelet Do..."]']


    # Testing function 1 : hf.get_value_list
    func_1_ok = False

    test_df["product_specifications"] = get_value_list(test_df["product_specifications"])
    func_1_ok = test_df.head(1)["product_specifications"][0] == \
    '[\'Elegance\', \'Door"}]\']'

    # Testing function 2 : hf.get_categories
    func_2_ok = False

    test_df = pd.concat([test_df,
                         get_categories(test_df["product_category_tree"])],
                        axis=1).drop(columns=["product_category_tree"])

    func_2_ok = test_df.head(1)["category"][0] == '["Home Furnishing ' \
                and\
                test_df.head(1)["subcategory"][0] == ' Curtains & Accessories '

    if func_1_ok and func_2_ok:
        return "All ok"
    elif func_1_ok:
        return "func 2 not ok"
    else:
        return "func 1 not ok"

    keypoints_df = keypoints_df.reset_index(drop=True)
    descriptors_df = descriptors_df.reset_index(drop=True)
    
    return (keypoints_df, descriptors_df)

#------------------------------------------

def predictWithText(product_name,
                    description,
                    product_specifications,
                    tf_idf_model,
                    prediction_model_type,
                    prediction_model):
    
    '''
        Returns the predicted category or the topic given the input
        text, the fitted tf-idf model, the prediciton / topic model,
        and model_type.

        Parameters
        ----------------
        - product_name           : string
                                   The name of the product

        - description            : string
                                   The name of the product
        
        - product_specifications : string
                                   The product specifications in the
                                   format :
                                   ['{"product_specification"=>[{"key"=>"Brand",
                                   "value"=>"Elegance"}, {"key"=>"Designed For",
                                   "value"=>"Door"}]}']
        
        - bow_model              : fitted TF-IDF model
        
        - prediction_model_type  : string
                                   The type of prediction_model
                                   ("LDA", "NMF", "MNB")
        
        - prediction_model       : fitted LDA, NMF or MNB model

        Returns
        ---------------
        - predicted   : the topic or category predicted by
                        the model for the given texts
    '''
    
    product = pd.DataFrame(columns=["product_name",
                                    "description",
                                    "product_specifications"])
    product.loc[len(product)] = [product_name,
                                 description,
                                 product_specifications]

    # 2. Clean text features
    product["product_specifications"] = get_value_list(product["product_specifications"])
    product = product.replace(r'^\s+$', np.nan, regex=True)
    product = product.replace(r'\n','', regex=True)
    product = product.replace(r'\r','', regex=True)
    product = product.replace(r'\t','', regex=True)
    product = product.replace(r'\[', '', regex=True)
    product = product.replace(r'\]', '', regex=True)
    product = product.replace(r'\{','', regex=True)
    product = product.replace(r'\}','', regex=True)
    product = product.replace(r'"','', regex=True)
    product = product.replace(r'\/',' ', regex=True)
    product = product.replace(r'\\','', regex=True)

    product["all_words"] = product["product_name"] + product["description"] + product["product_specifications"]

    # 3. Prétraitement
    # Passage en minuscules
    product["all_words"] = product["all_words"].str.lower()
    product["all_words"] = product["all_words"].str.rstrip()
    product["all_words"] = product["all_words"].str.lstrip()

    # Tokenization
    tokenizer = nltk.RegexpTokenizer(r'[a-zA-Z]+')
    product["all_words"] = product["all_words"].apply(lambda x: tokenizer.tokenize(x))

    # Deleting stopwords
    product["all_words"] = product["all_words"].apply(lambda x: [word \
                                                                 for word \
                                                                 in x \
                                                                 if word \
                                                                 not in stopwords.words('english')])

    # Lemmatization
    product["all_words"] = product["all_words"].apply(lambda x: lemmatized_list(x))

    # 4. Transform to bow
    text_tf_idf = tf_idf_model.transform(product["all_words"].astype("str"))
    
    #5 Predict
    if(prediction_model_type == "LDA" or prediction_model_type == "NMF"):
        predicted = prediction_model.transform(text_tf_idf)
    else:
        predicted = (prediction_model.predict(text_tf_idf),
                     prediction_model.predict_proba(text_tf_idf))
    
    return predicted

#------------------------------------------

def predictWithImage(image_path, visual_words, model):
    '''
        Returns the predicted category for the input
        image at image_path, the dictionary of visual word,
        and the fitted model.

        Parameters
        ----------------
        - image_path       : string
                             The path of the image whose category
                             will be predicted

        - visual_words     : string
                             The name of the product
        
        - prediction_model : fitted model

        Returns
        ---------------
        - predicted   : the category predicted by
                        the model for the given image at image+path
    '''
    
    # 1. Load image
    img = Image.open(image_path)

    # 2. Correct histogram
    corrected_img = ImageOps.equalize(ImageOps.autocontrast(img))

    # 3. Apply gaussian filter
    corrected_gauss_img = corrected_img.filter(ImageFilter.GaussianBlur(radius=3))

    # 4. Get keypoints and descriptors
    KEYPOINTS = 10

    img_df = pd.DataFrame(columns=["corrected_gauss_img",
                                   "image",
                                   "category",
                                   "subcategory"])

    img_df.loc[len(img_df)] = [corrected_gauss_img,
                               "test_picture.jpg",
                               "TBP",
                               "TBP"]

    keypoints_gauss, descriptors_gauss = assembleKeypointsDescriptors(img_df,
                                                                      "corrected_gauss_img",
                                                                       KEYPOINTS)

    # 5. Get bovw
    bovw_gauss = getImgsHisto(descriptors_gauss, visual_words)


    # 6. Give it to knn
    return model.predict(bovw_gauss.iloc[:,:-3]),model.predict_proba(bovw_gauss.iloc[:,:-3])

#------------------------------------------

def get_categories_per_topic(n_topics, model, text_data, product_data):
    '''
        Returns the proportion of products from each category
        for each topic.
        
        Parameters
        ----------------
        - n_topics     : int
                         Number of topics
        
        - model        : fitted model
        - text_data    : pandas dataframe
                         input data
        - product_data : pandas dataframe
                         the dataframe containing the
                         original data, including the
                         categories

        Returns
        ---------------
        - categories_topics : pandas dataframe
                              the proportion of products
                              from each category for each topic
    '''
    
    doc_topic = model.transform(text_data)
    
    products_topic = product_data.copy()
    
    topics = []
    
    for n in range(doc_topic.shape[0]):
        topic_most_pr = doc_topic[n].argmax()
        topics.append(topic_most_pr)
        
    products_topic["topics"] = topics
    
    percent_predicted = pd.DataFrame(columns=["Topic",
                                          "% Home Furnishing",
                                          "% Baby Care",
                                          "% Watches",
                                          "% Home Decor & Festive Needs",
                                          "% Kitchen & Dining",
                                          "% Beauty and Personal Care",
                                          "% Computers"])

    for topic, data_topic in products_topic.groupby("topics"):
        
        row = [topic,
               (len(data_topic[data_topic["category"]=="Home Furnishing"])*100)/len(data_topic),
               (len(data_topic[data_topic["category"]=="Baby Care"])*100)/len(data_topic),
               (len(data_topic[data_topic["category"]=="Watches"])*100)/len(data_topic),
               (len(data_topic[data_topic["category"]=="Home Decor & Festive Needs"])*100)/len(data_topic),
               (len(data_topic[data_topic["category"]=="Kitchen & Dining"])*100)/len(data_topic),
               (len(data_topic[data_topic["category"]=="Beauty and Personal Care"])*100)/len(data_topic),
               (len(data_topic[data_topic["category"]=="Computers"])*100)/len(data_topic)]
        
        percent_predicted.loc[len(percent_predicted)] = row
    
    percent_predicted = percent_predicted.set_index("Topic")

    return percent_predicted

#------------------------------------------

def generate_visual_word_pics(visual_words, data_clustering, feature_img, keypoints, descriptors, directory):
    '''
        Creates the visual word pictures by plotting the image and
        the keypoints using the keypoints directory and size.
        
        The created pictures are stored in the following directories :
        
        - output/cropped_pictures/directory for the visual_word alone
        - output/full_pictures/directory for the full picture, with the
          keypoint drawn on.
        
        Parameters
        ----------------
        - visual_words    : array
                            the centroïdes of the clustering
        
        - data_clustering : pandas dataframe
                            data used for the clustering
                            
        - feature_img     : string
                            the name of the feature containing
                            the images containing the keypoints
                            
        - keypoints       : pandas dataframe
                            keypoints
        - descriptors     : pandas dataframe
                            descriptors
                            
        - directory       : string
                            the subdirectory's name to store the
                            pictures in

        Returns
        ---------------
        -
    '''
    
    # Find the medoids
    closest, _ = pairwise_distances_argmin_min(visual_words,
                                               data_clustering.iloc[:, :-1])

    data_medoids = pd.concat([data_clustering.loc[list(closest)],
                              descriptors.loc[list(closest)],
                              keypoints.loc[list(closest)][[0]].rename(columns={0:"keypoints"})],
                          axis=1)
                          
     
    # For each medoid :
    # make a picture of the keypoint (a square around the center of the keypoint
    # with size =  keypoint size / 2 )
    # make a picture of the picture with the traced contour of the keypoint
    
    for medoid in range(len(data_medoids)):
        img = data_medoids.iloc[medoid][feature_img]
        kp = data_medoids.iloc[medoid]["keypoints"]
        
        if directory == "":
            cropped_pics_path = "output/cropped_pictures/cropped_"+str(medoid)+".png"
            full_pics_path = "output/full_pictures/medoid_"+str(medoid)+".png"
        else:
            cropped_pics_path = "output/cropped_pictures/"+directory+"/cropped_"+str(medoid)+".png"
            full_pics_path = "output/full_pictures/"+directory+"/medoid_"+str(medoid)+".png"

        # transform to opencv image
        opencv_img = np.array(img)[:, :, ::-1].copy()
        
        # create new cropped image
        y = int(kp.pt[1])
        x = int(kp.pt[0])
        c = kp.size

        Y = int(y+kp.size)
        X = int(x+kp.size)

        crop_img = opencv_img[y:Y, x:X]
        cv.imwrite(cropped_pics_path, crop_img)
        
        # create new image
        image = cv.rectangle(opencv_img,
                         (int(kp.pt[0]-kp.size/2), int(kp.pt[1]-kp.size/2)),
                         (int(kp.pt[0]+kp.size/2), int(kp.pt[1]+kp.size/2)),
                         (255,0,0), 2)
        
        cv.imwrite(full_pics_path, image)
