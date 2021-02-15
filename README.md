# **Automatic Classification of Goods: A Feasability Study**
*Sofia Chevrolat (August 2020)*

> NB: This project is the sixth of a series of seven comprising [the syllabus offered by OpenClassrooms in partnership with Centrale Supélec and sanctioned by the Data Scientist diploma - Master level](https://openclassrooms.com/fr/paths/164-data-scientist).
___

This study aims to study the feasability of a classication motor for goods of different categories, based on their photo and description.
This classification should be precise enough to fully automatize the categorization of articles on an e-commerce website when seller upload their item.
___

## Table of Contents

This notebook is organized as follows:

**1. Setting Up**
- 1.1 Loading the necessary libraries and functions
- 1.2 Loading and description of the data set
    * 1.2.1 Description of the data set
    * 1.2.2 Visualization of the features' completion rate
- 1.3 Data targeting
- 1.4 Unpacking of the product specification and categories
- 1.5 Concatenating product features (categories excluded)
- 1.6 Description of the final data set

**2. Pre-treatment**
- 2.1 Text data
    * 2.1.1 Deleting special characters and unnecessary spaces
    * 2.1.2 Transformation to lowercase
    * 2.1.3 Tokenization
    * 2.1.4 Deleting stopwords
    * 2.1.5 Lemmatization
- 2.2 Image data 
    * 2.2.1 Correcting exposure: histogram stretching
        * 2.2.1.1 Visualization: pictures before correction
        * 2.2.1.2 Histogram stretching
        * 2.2.1.3 Visualization: pictures after correction
    * 2.2.2 Correcting contrast: histogram equalization
        * 2.2.2.1 Visualization: pictures before correction
        * 2.2.2.2 Histogram equalization
        * 2.2.2.3 Visualization: pictures after correction
    * 2.2.3 Correcting noise : filtering 
        * 2.2.3.1 Gaussian filtering
        * 2.2.3.2 Median filtering
    * 2.2.4 Comparison before - after corrections
        
**3. Exploratory Analysis**
- 3.1 Category and subcategory distribution
    * 3.1.1 Category distribution
    * 3.1.2 Subcategory distribution per category
- 3.2 Word distribution
- 3.3 Word frequency
    * 3.3.1 In product names per category (<i>product_name</i>)
    * 3.3.2 In product descriptions per category (<i>descriptions</i>)
    * 3.3.3 In product specs per category (<i>product_specifications</i>)
    * 3.3.4 In all text features together (<i>all_words</i>)
    
**4. Feasability Study**
- 4.1 Using image data
    * 4.1.1 Extracting key points and their descriptors
    * 4.1.2 Creating the visual word dictionary
        * 4.1.2.1 Clustering
        * 4.1.2.2 Visualization: a few visual words
    * 4.1.3 Creating BOVW for all the pictures
        * 4.1.3.1 Calculating the BOVW
        * 4.1.3.2 Visualization: a few BOVW / category
    * 4.1.4 Prediction tests
        * 4.1.4.1 On the data set
        * 4.1.4.2 On a product outside the data set
    * 4.1.5 Conclusions
- 4.2 Using text data
    * 4.2.1 Extracting the features
        * 4.2.1.1 Transformation into a BOW matrix
        * 4.2.1.2 Transformation into a TF-IDF matrix
    * 4.2.2 Dimensionality reduction
        * 4.2.2.1 PCA
        * 4.2.2.2 t-SNE
        * 4.2.2.3 Comparison
    * 4.2.3 Clustering
        * 4.2.3.1 Creating the model
        * 4.2.3.2 Using the model
    * 4.2.4 Prediction tests
        * 4.2.4.1 On the data set
        * 4.2.4.2 On a product outside the data set
    * 4.2.5 Conclusions

**5. Conclusion**
_________

## Requirements

This assumes that you already have an environment allowing you to run Jupyter notebooks. 

The libraries used otherwise are listed in requirements.txt

_________

## Usage

1. Download the dataset from [OpenClassrooms](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Textimage+DAS+V2/Dataset+projet+prétraitement+textes+images.zip), and place:
- the text data under Resources/datasets/
- the image data under Resources/pictures/

2. Run the following in your terminal to install all required libraries :

```bash
pip3 install -r requirements.txt
```

4. Run the notebook.
__________

## Results

For a complete presentation and commentary of the results of this analysis, please see the PowerPoint presentation.

> NB: The presentation is in French.
