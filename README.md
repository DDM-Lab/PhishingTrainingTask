# Phishing Training Task

Singh, K., Aggarwal, P., Rajivan, P., & Gonzalez, C. (2019, November). Training to detect phishing emails: Effects of the frequency of experienced phishing emails. In Proceedings of the human factors and ergonomics society annual meeting (Vol. 63, No. 1, pp. 453-457). Sage CA: Los Angeles, CA: SAGE Publications.


The Phishing Training Task is designed to help participants identify Phishing emails. The task consists of three phases: pre-training, training, and post-training, during which each participant will examine 60 emails randomly selected from a dataset of 239 emails. Specifically, participants will review 40 emails during the training phase and 10 emails (8 Ham and 2 Phishing), each during the pre-training and post-training phases. Plus, during the pre- and post-training phases, participants won't receive any feedback on whether their decision was correct. In contrast, in the training phase, they will receive feedback regarding the correctness of their choices.
The human dataset had 297 participants divided evenly into three groups. Each group received a different percentage of Phishing emails during the training phase: low (10/40), medium (20/40), and high (30/40).


The Instance-Base Learning (IBL) model used in this project was developed initialy by [Palvi Aggarwal](https://github.com/palvi-12) and [Shova Kuikel](https://github.com/Shovaa) and later on incremented to include model-fitting and model-tracing by [Maria José Ferreira](https://github.com/MariaJoseF).



## Data files

The emails dataset and the human experiment dataset are available on OSF: https://osf.io/r83ag/

For this particular implementation we are using:
- PhisingDataset_HFES2019.csv
- experiment1-outcomefeedback.csv 
- DataInformation.txt (dictionary with variables information)

`max_decays_M1.csv` and `max_decays_M2.csv` were generated after running Model-Fitting and the R script for each model and they contain the best decay for each participant.

`similarity_matrix_emails_nli_large.csv`, `similarity_matrix_sender_nli_large.csv` and `similarity_matrix_subject_nli_large.csv` were generated for each of the features used in our IBL model using a pre-trained language model of Bidirectional Encoder Representation of Transformer (BERT). The development of these similarity calculations was done by [Palvi Aggarwal](https://github.com/palvi-12) and [Shova Kuikel](https://github.com/Shovaa).

**NOTE:** For convenience, we have added the CSV files to the `Data` folder.

## A tour of the code

- **`tracingModel1.py`** and **`tracingModel2.py`** contain the implementation of the IBL models for **Model-Fitting** and **Model-Tracing**, each with a different training size. `Model 1` is trained with 10 emails from the pre-training phase and predicts the remaining 50 from the training and post-training phases. In contrast, `Model 2` is trained with 50 emails from the pre-training and training phases and predicts the last 10 from the post-training phase.
    - To run this code for each training model, you will need the files `experiment1-outcomefeedback.csv`, `similarity_matrix_emails_nli_large.csv`, `similarity_matrix_sender_nli_large.csv` and `similarity_matrix_subject_nli_large.csv` located in the `Data` folder.
    
    - To run **`Model-Fitting`**, search for the **\_\_main\_\_** function (at the end of the script), uncomment the function **`runModelFitting`**, and comment all the other functions.
        - These scripts will generate 291 IBL models each with a `decay value` in the range of `[0.1 - 3]` with increments of `0.01` for each participant data.
        - These scripts will generate automaticaly:
            - The **fitting results** of all the models per participant and record it as `Tracing_Data_Fitting_{ID}.csv` in the folders `Generated_Models_Data\Tracing_Results_Fitting_{ID}` on the root of the project. 
            - The **fitting metric (`SyncRate`) results** per model for each participant and record it as `MaxSyncRate_Fitting_Data_{ID}.csv` in the folder `Generated_Models_Data` on the root of the project. 
                - `SyncRate` examines the synchronization between the model prediction and each human choice. We determined whether the model prediction was the same as the actual human action for each decision. If it was the same, the synchronization value for that decision was `1`; otherwise, it was `0`. 
                - We calculated the `SyncRate average` per participant for each IBL model (with a different decay value) for the total number of decisions each model in each training settings made. In the case of `Model 1` the total of decisions is 50 (40 from training and 10 from post-training). While in `Model 2` the total of decisions is 10 (post-training data). If multiple models have the same `SyncRate`, the model with the highest decay is selected for that participant.
            - `{ID}` is replaced by `M1` or `M2` according to the training script selected.

    - To run **`Model-Tracing`** with personalized decay per participant you will need the file `max_decays_{ID}.csv` generated by the **`Phishing.Rmd`** script (for convenience there is a copy in the `Data` folder). `{ID}` is replaced by `M1` or `M2` according to the training script selected.
        - Search for the **\_\_main\_\_** function (at the end of the script), uncomment the function **`StartTracing`** with the following parameters: `tracing=True`, `decay=0`, `best_decayFile=best_decay`, `fitting=False`, and comment all the other functions. `best_decay` will contain the `decay value` that best fit each participant data recorded in `max_decays_{ID}.csv`.
        - These scripts will generate automaticaly the tracing results with models runing the best decay per participant and record it as `Tracing_Data_Personalized_{ID}.csv` in the folders `Generated_Models_Data\Tracing_Results_{ID}` on the root of the project.
 
    **NOTE:**  For convenience the **\_\_main\_\_** function has comments regarding each setting and what parameters manipulations need to be run.

- **`confusionMatric.py`** contains functions to generate confusion matrixes and learning curves in different settings.
    - To run this code, you will need the files `Tracing_Data_Personalized_M1.csv` and `Tracing_Data_Personalized_M2.csv` generated by the `tracingModel1.py` and `tracingModel2.py` scripts available in the folders `Generated_Models_Data\Tracing_Results_{ID}`. `{ID}` is replaced by `M1` or `M2` according to the training script selected.
    - This script will generate automatically images in the `PNG` format with confusion matrixes and learning curves of the `Model Predictions` and/or the `Human Choices` in different settings. These images will be automaticaly stored in a folder named `Plots_{path}\Individual` in the root of the project. `{path}` is replaced by either `Tracing_ConfusionMatrixes` or `Tracing_LearningCurves` depending on the settings analysed.

- **`Phishing.Rmd`** is an `R` code script used to assess the best fitting decay for each participant based on the 291 Models generated.
    - To run this script, you first must run the `tracingModel1.py` and `tracingModel2.py` with **`Model-Fitting`** and use the generated `MaxSyncRate_Fitting_Data_M1.csv` and `MaxSyncRate_Fitting_Data_M2.csv`files located in the folder `Generated_Models_Data`.
    - This script will automatically generate the `max_decays_M1.csv` and `max_decays_M2.csv` in the same folder where the script is located (same files as the ones in the `Data` folder).

    **NOTE:** The `R script` was run in `Rstudio 2024.04.2+764` and needs an independent project. A copy of the files `MaxSyncRate_Fitting_Data_M1.csv` and `MaxSyncRate_Fitting_Data_M2.csv` needs to be added inside a folder named `Data\Phishing` in the root of the R project. After running this script, the generated files need to be copied into the Python project in the `Data` folder.

## Installation

Ensure Python version 3.8 or later is installed.

Run `python3 -m venv venv` or `python -m venv venv` to create a virtual environment.

Run `source venv/bin/activate` or `.\venv\Scripts\activate` to activate the virtual environment.

Run `pip install -r requirements.txt` to install the required Python packages into the virtual environment.

**NOTE:** If you do not want to create a virtual enviroment, just run `pip install -r requirements.txt`.


## More information

For more information or details, please contact [ddmlab@andrew.cmu.edu](mailto:ddmlab@andrew.cmu.edu).