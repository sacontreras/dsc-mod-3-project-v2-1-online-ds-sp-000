
# Module 3 Final Project - Directory Contents

## [project-overview.ipynb](project-overview.ipynb)
This notebook provides a high-level overview of my project.  Open this notebook first.


## [project-workflow.graphml](project-workflow.graphml)
This is simply the yEd (diagramming software) project file used to develop the project-overview diagram.  It is only included to track progress in Github.


## [project-workflow.png](project-workflow.png)
This is simply the final image of the finalized project-overview diagram exported from the [project-workflow.graphml](project-workflow.graphml) yEd project.


## [dask-cloud-setup.md](dask-cloud-setup.md)
This file contains the steps required to set up a Kubernetes Cluster in Google Cloud Compute in order to use Remote Dask Parallel Data Processing.

Dask usage within this project has been depracated since, in order to fully utilize it in order to truly achieve Data Parallelism, would require a near FULL refactoring of the entire project.  

This would be a good candidate for future work.


## [EDA.ipynb](EDA.ipynb)
This notebook contains all Exploratory Data Analysis and results thereof.


## [eda-config.txt](eda-config.txt)
The contents of this file are used by [EDA.ipynb](EDA.ipynb).

It contains the following information:
1. structured information (URL, local filename, etc) required to retrieve data files (labeled predictors, labels, and unlabeled predictors) from a remote source
2. feature groups - features are grouped/related by their provided descriptions (from [https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/25/](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/25/))


## [preprocessing.ipynb](preprocessing.ipynb)
This notebook implements preprocessing steps required for model-building, resulting from in-depth analysis conducted within [EDA.ipynb](EDA.ipynb).  After the execution of this notebook is complete, the resulting output is a JSON file containing the list of "best" preprocessing options (that should produce models with the highest validation accuracies), saved to file "preprocessing-results/preprocessing-spec-last.json" - that file is used as input to, and is one of (configuration) files necessary that drive the execution of, [models.ipynb](models.ipynb).


## [preprocessing-config.txt](preprocessing-config.txt)
This is the file containing all possible preprocessing options for each feature within each feature-group.  It is used as input which drives the algorithm (within [preprocessing.ipynb](preprocessing.ipynb)) that selects the "best" preprocessing options in order to "evolve" the best models (with highest validation accuracy).  The nature of and logic involved in which preprocessing options are applicable to a given feature (the contents of this file) is developed in the [EDA.ipynb](EDA.ipynb) notebook.


## [models.ipynb](models.ipynb)
The purpose of this notebook, in the end, is to build models based on the optimized selection (done within [preprocessing.ipynb](preprocessing.ipynb)) of applicable preprocessing options "discovered" within [EDA.ipynb](EDA.ipynb).

This notebook build models of associated (multi) classification algorithms configured within [models-config.txt](models-config.txt) (that file is required input for this notebook).

This notebook also uses "preprocessing-results/preprocessing-spec-last.json" (which is output as the result of executing [preprocessing.ipynb](preprocessing.ipynb) through to conclusion) as input.  This is because it is this notebook that applies those (best) preprocessing transformation options in order to produce the actual transformed labeled (split into training and validation partitions) and unalabeled data sets, which are then finally used to reach our ultimate goal: to train models and make predictions from unalabeled (transformed) predictors.


## [models-config.txt](models-config.txt)
This file contains configuration parameters that drive the entire model building process that occurs within [models.ipynb](models.ipynb).


## [TFIDF-KMeans-CategoricalClassification-Algo.ipynb](TFIDF-KMeans-CategoricalClassification-Algo.ipynb)
This is the notebook which houses the research conducted to develop the experimental mental to use as an alternative to OneHot-Encoding HIGH-CARDINALITY categorical variables.  The explanation of this technique is beyond the scope of this simple readme file.  Please refer to [TFIDF-KMeans-CategoricalClassification-Algo.ipynb](TFIDF-KMeans-CategoricalClassification-Algo.ipynb) if you're interested.  

Note that the TF-IDF KMeans Classification technique has been demonstrated to improve accuracy compared to One-Hot Encoding!  BUT... it turns out that including any of the three high-categorical features (to which this technique applies) DRASTICALLY decreases accuracy overall - by a ridiculous amount... almost a 30% drop in accuracy - NO MATTER the technique used.  To be clear, One-Hot Encoding drops accuracy the most... TF-IDF KMeans Classification drops accuracy less (but still by a significant amount).  Therefore, since it has been shown that no qualifying (high-cardinality) categorical feature is useful to modeling (just the opposite) under any circumstance or technique, THIS technique has been depracated for this project entirely.

HOWEVER, the base technique of TD-IDF Normalization, which is a building block required for TF-IDF-Kmeans Classification, IS shown to be useful in model building.  The base technique is described in this notebook ([TFIDF-KMeans-CategoricalClassification-Algo.ipynb](TFIDF-KMeans-CategoricalClassification-Algo.ipynb)), which is why it still remains in the project.

Further development of this experimental technique/method would also be an excellent candidate for future work.


## Data Files
Since some (RAW) data files are rather large (20 MB+), they have not been checked into Github.  However, each notebook has the routines necessary to download required files as necessary and they will obviously remain in this directory after that.
