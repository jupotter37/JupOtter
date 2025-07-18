# JupOtter: Cell-Level Bug Detection in Jupyter Notebooks. Supplementary material


### How to run JupOtter:
- 'run_model.ipynb' is used to run JupOtter or tokenize content for training. This is where JupOtter can be tested on notebooks, it contains code for file-level, cell-level, and single notebook evaluation.
- 'train_Otter.ipynb': can be used to train JupOtter, it assumes tokenized content from 'run_model.ipynb'. 


### How to run our Flake8 configurations:
- 'Flake8' directory contains the code used for Flake8 file-level and cell-level evaluation. These are set up for testing, the file-level and cell-level testing files assume tokenized data from 'run_model.ipynb'. This was done to ensure JupOtter and Flake8 were tested on the exact same data.

### Datasets:
- OtterDataset is organized in directories by query used, all queries are included in the OtterDataset folder and were used between December 2024 and January 2025. The final labeled 'OtterDataset.csv' is available in 'dataset\OtterDataset\OtterDataset.csv'. Note that some notebooks retrieved using our queries were filtered out during data processing.
- Our subset of the CodeParrot GitHub Jupyter Notebooks dataset used during evaluation is included in 'dataset\CodeParrot_Subset'.
- 'errorExamples_Figure6_Figure8' contains notebook files referenced in figures whos repositories have since been removed from GitHub.
- Also included are the files used to process code which convert directories of notebooks into a labeled dataset of notebooks ready for training and evaluation. These include 'bug_info_proccesing.py', 'code_proccessing.py', and cleanData_csv.py'. These files are used within 'run_model.ipynb', cell 2 to prepare data for tokenization. These files were used to process the directories of files within OtterDataset, first creating an individual CSV datasets for each directory, then combining them all while ensuring duplicates are removed. To create OtterDataset, the function 'create_notebook_train_data()' from the 'clean()' class in 'cleanData_csv.py' was used to create csv files for each directory, then 'moveBuggyCode()' and 'moveCode()' were used from the 'Move_books()' class to combine them and remove duplicates.
- 'dataset\download_data' contains the python used to download our external datasets.

### Evaluation results
- The training and evaluation results for JupOtter-small and JupOtter-base are included in the 'trainin_and_evaluation_results' directory. Validation testing within training epochs was done using the OtterDataset test set.
- Evaluation results for Flake8 configured for both file-level and cell-level bug detection are also included under 'trainin_and_evaluation_results'.
- Trained model parameters for the JupOtter-small and JupOtter-base epochs used durring evaluation are included in the 'models' directory.



This repository is 20.2 GB due to the size of the raw notebook files, labeled datasets, and model parameters included.






