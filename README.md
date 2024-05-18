# Phishing Training Task

Singh, K., Aggarwal, P., Rajivan, P., & Gonzalez, C. (2019, November). Training to detect phishing emails: Effects of the frequency of experienced phishing emails. In Proceedings of the human factors and ergonomics society annual meeting (Vol. 63, No. 1, pp. 453-457). Sage CA: Los Angeles, CA: SAGE Publications.

## Data files

The emails dataset and the human experiment dataset are available on OSF: https://osf.io/r83ag/

For this particular implementation we are using:
    - PhisingDataset_HFES2019.csv
    - experiment1-outcomefeedback.csv 
    - DataInformation.txt (dictionary with variables information)

max_decays_M1.csv and max_decays_M2.csv were generated after running Model-Fitting for each model and they contain the best decay for each participant.



## IBL model

The IBL model was developed initialy by Palvi Aggarwal and Shova Kuikel and later on incremented to include model-fitting and model-tracing by Maria José Ferreira.

confusionMatric.py contains functions to generate confusion matrixes and Learning curves for both models in different settings.

tracingModel1.py and tracingModel2.pt have the developped IBL model.
