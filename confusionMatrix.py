from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime

import os
import sys

import math
import matplotlib .pyplot as plt
import csv
import pandas as pd
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(script_dir,"Generated_Models_Data")



def GenerateConfusionMatrix(actual_decisions, predicted_decisions, label, tracing, model, path):
  
    # Generate current date and time
    current_datetime = datetime.now().strftime('%Y_%m_%d_%H_%M')

    if tracing:
        file_path = os.path.join(script_dir, f"Plots_{path}\\Individual")
        plotName = f"CM_Tracing_{label}_{model}_{current_datetime}.png"
    else:
        file_path = os.path.join(script_dir, f"Plots_{path}\\Individual")
        
        plotName = f"CM_NoTracing_{label}_{model}_{current_datetime}.png"
    os.makedirs(file_path, exist_ok=True)

    plot_path = os.path.join(file_path,plotName)

    # Construct the confusion matrix
    cm = confusion_matrix(actual_decisions, predicted_decisions, labels=["Phishing", "Ham"], normalize='all')


    # Print the confusion matrix
    print("Confusion Matrix:")
    print(cm)

    # Extract values from confusion matrix
    true_positives = cm[0, 0]  # Phishing correctly predicted as Phishing
    true_negatives = cm[1, 1]  # Ham correctly predicted as Ham
    total_instances = np.sum(cm)  # Sum of all instances

    annot = np.empty_like(cm).astype(str)

    labelsCM = [['Hits', 'False Alarm'],
          ['Misses', 'Correct Rejection']]

    for i in range(2):
        for j in range(2):
            annot[i, j] =labelsCM[i][j]


    # Calculate accuracy
    accuracy = round((true_positives + true_negatives) / total_instances,2)

    print(f'Accuracy: {accuracy}')



    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Phishing", "Ham"])
    #disp.plot().figure_.savefig(f'{plot_path}')
    disp.plot()

    ax = disp.ax_

    #Adding labels to the four positions
    for i, text in enumerate(ax.texts):
    # Modify the text as needed
        x = text.get_position()[0]
        y= text.get_position()[1]
        text_str = text.get_text()

        # Convert the string to a float
        float_value = float(text_str)

        # Truncate the float to 2 decimal places
        truncated_float_value = int(float_value * 100) / 100 #float("{:.2f}".format(float_value))
        

    
        new_text = f"{labelsCM[x][y]}\n{truncated_float_value}"
        text.set_text(new_text)


    if "Phishing" in label or "Ham" in label:
        plt.xlabel("Model Predictions")  # Label for x-axis
        plt.ylabel("Human Choices")  # Label for y-axis
        plt.title(f"Ground Truth = {label}\n Accuracy = {accuracy}, {model}")  # Title of the plot
    elif "Human-Model" in label:
        plt.xlabel("Model Predictions")  # Label for x-axis
        plt.ylabel("Human Choices")  # Label for y-axis
        plt.title(f"Syncronization {label}\n Accuracy = {accuracy}, {model}")   # Title of the plot
    else:
        if "Human" in label:
            plt.xlabel("Human Choices")  # Label for x-axis
            plt.title(f"{label} \n Accuracy = {accuracy}")  # Title of the plot
        else:
            plt.xlabel("Model Predictions")  # Label for x-axis
            plt.title(f"{label} \n Accuracy = {accuracy}, {model}")  # Title of the plot
        plt.ylabel("Ground Truth")  # Label for y-axis
        
    plt.savefig(f'{plot_path}')  # or .jpg, .pdf, etc.
    plt.show()
    plt.close()

    return plot_path
    


def GenerateLearningCurves(predicted_decisions, label, tracing, model, path):
  
    # Generate current date and time
    current_datetime = datetime.now().strftime('%Y_%m_%d_%H_%M')

    if tracing:
        file_path = os.path.join(script_dir, f"Plots_{path}\\Individual")
        plotName = f"LC_Tracing_{label}_{model}_{current_datetime}.png"
    else:
        file_path = os.path.join(script_dir, f"Plots_{path}\\Individual")
        plotName = f"LC_NoTracing_{label}_{model}_{current_datetime}.png"
    os.makedirs(file_path, exist_ok=True)

    plot_path = os.path.join(file_path,plotName)

    plt.figure()  # Create a new figure

    
    # Convert 'Trial.number' column to numeric, coercing errors to NaN
    predicted_decisions['Trial.number'] = pd.to_numeric(predicted_decisions['Trial.number'], errors='coerce')

    # Add 40 to Trial.number where Training.Phase is 3
    predicted_decisions.loc[predicted_decisions['Training.Phase'] == 3, 'Trial.number'] += 40

    accuracy = predicted_decisions.groupby(["Trial.number", "Training.Phase"]).agg(
    avg_modelSyncUser=('modelSyncUser', 'mean'),
    ).reset_index()


    predicted_decisions.loc[predicted_decisions['Training.Phase'] == 3, 'Trial.number'] -= 40

    plt.plot(accuracy['Trial.number'], accuracy['avg_modelSyncUser'])
        
    
    # Set y-axis range between 0 and 1
    plt.ylim(0, 1)

    # Add vertical dashed line at trial 40
    plt.axvline(x=40, color='grey', linestyle='--')
    # Add horizontal dashed line at y=0.5 with grey color
    plt.axhline(y=0.5, color='grey', linestyle='--')
    
    plt.xlabel("Trials")  # Label for x-axis
    plt.ylabel("Average of SyncRate")  # Label for y-axis

    if "Phishing" in label or "Ham" in label:
        plt.title(f"Learning Curve \n Ground Truth = {label}\n {model}")  # Title of the plot
    elif "Human-Model" in label:
        plt.title(f"Learning Curve\n Syncronization {label}\n {model}")   # Title of the plot
    else:
        if "Human" in label:
            plt.title(f"Learning Curve {label}\n")  # Title of the plot
        else:
            plt.title(f"Learning Curve {label}\n {model}")  # Title of the plot
    
    plt.tight_layout()

    plt.savefig(f'{plot_path}')  # or .jpg, .pdf, etc.
    plt.show()
    plt.close()

    return plot_path

def CombinePlots(dataPlots, rows, cols, title, tracing, model, label, path):

    # Generate current date and time
    current_datetime = datetime.now().strftime('%Y_%m_%d_%H_%M')

    if tracing:
        file_path = os.path.join(script_dir, f"Plots_{path}\\Grouped")
        if "ConfusionMatrixes" in path:
            plotName = f"CM_Tracing_{label}_{model}_{current_datetime}.png"
        else:
            plotName = f"LC_Tracing_{label}_{model}_{current_datetime}.png"
    else:
        file_path = os.path.join(script_dir, f"Plots_No{path}\\Grouped")
        if "ConfusionMatrixes" in path:
            plotName = f"CM_NoTracing_{label}_{model}_{current_datetime}.png"
        else:
            plotName = f"LC_NoTracing_{label}_{model}_{current_datetime}.png"
    os.makedirs(file_path, exist_ok=True)

    plot_path = os.path.join(file_path,plotName)

    # Combine plots
    if rows == 1:

        fig, axes = plt.subplots(rows, cols, figsize=(6, 4))
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(8, 7))

    for i, plot in enumerate(dataPlots):
        row = i // cols
        col = i % cols
        if rows == 1:
            img = plt.imread(plot)
            axes[col].imshow(img)
            axes[col].axis('off')  # Hide axes for individual plots
        else:
            img = plt.imread(plot)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')  # Hide axes for individual plots
        

   # Set title for the entire plot
    plt.suptitle(f'{title}')
    plt.tight_layout()  # Adjust layout
    plt.savefig(f'{plot_path}')  # or .jpg, .pdf, etc.
    plt.show()
    plt.close()

if __name__ == "__main__":

    dataFile = ["Tracing_Results_M1\Tracing_Data_Personalized_M1.csv", "Tracing_Results_M2\Tracing_Data_Personalized_M2.csv"]

    plotsStep1_CM = []
    plotsStep2_CM = []
    plotsStep3_CM = []

    plotsStep1_LC = []
    plotsStep2_LC = []
    plotsStep3_LC = []

    for f in dataFile:

        file_path = os.path.join(DATA_FILE, f)

        predictedData = pd.read_csv(file_path)
        phishing_data = []
        ham_data = []
        PersonalizedData = []

        if "No_Tracing" in f:
            tracing = False
        else:
            tracing = True

        if "M1" in f:
            # Filter data where 'Ground Truth' is 'Phishing'
            phishing_data = predictedData[(predictedData['ground.truth'] == 'Phishing') & 
                                        ((predictedData['Training.Phase'] == 2) |
                                            (predictedData['Training.Phase'] == 3))]
            
            # Filter data where 'Ground Truth' is 'Ham'
            ham_data = predictedData[(predictedData['ground.truth'] == 'Ham') & 
                                        ((predictedData['Training.Phase'] == 2) |
                                            (predictedData['Training.Phase'] == 3))]
            

            PersonalizedData = predictedData[((predictedData['Training.Phase'] == 2) |
                                            (predictedData['Training.Phase'] == 3))]

            model_type="Training1"
        else:
            # Filter data where 'Ground Truth' is 'Phishing'
            phishing_data = predictedData[(predictedData['ground.truth'] == 'Phishing') & 
                                            (predictedData['Training.Phase'] == 3)]
            
            # Filter data where 'Ground Truth' is 'Ham'
            ham_data = predictedData[(predictedData['ground.truth'] == 'Ham') & 
                                            (predictedData['Training.Phase'] == 3)]
            
            PersonalizedData = predictedData[(predictedData['Training.Phase'] == 3)]

            model_type="Training2"


        #################### Plot Model-Human Data  all training phases together

        human_predicted =  np.array(PersonalizedData['user.choice.made.decision']).astype(str)
        model_predicted  = np.array(PersonalizedData['model.choice.made.decision']).astype(str)
        plot_CM = GenerateConfusionMatrix(human_predicted , model_predicted ,'Human-Model',tracing, model_type, "Tracing_ConfusionMatrixes")
        plot_LC = GenerateLearningCurves(PersonalizedData ,'Human-Model',tracing, model_type, "Tracing_LearningCurves")

        plotsStep1_CM.append(plot_CM)
        plotsStep1_LC.append(plot_LC)

        
        #################### All training phases together


        # Filter data where 'Ground Truth' is 'Phishing' 

        phishingHuman =  np.array(phishing_data['user.choice.made.decision']).astype(str)
        phishingModel = np.array(phishing_data['model.choice.made.decision']).astype(str)

        plot_CM = GenerateConfusionMatrix(phishingHuman, phishingModel, 'Phishing', tracing, model_type, "Tracing_ConfusionMatrixes")
        plot_LC = GenerateLearningCurves(phishing_data, 'Phishing', tracing, model_type, "Tracing_LearningCurves")

        phishing_data['Trial.number']
        plotsStep2_CM.append(plot_CM)
        plotsStep2_LC.append(plot_LC)
        
        # Filter data where 'Ground Truth' is 'Ham'

        hamHuman =  np.array(ham_data['user.choice.made.decision']).astype(str)
        hamModel = np.array(ham_data['model.choice.made.decision']).astype(str)

        plot_CM = GenerateConfusionMatrix(hamHuman, hamModel, 'Ham', tracing, model_type, "Tracing_ConfusionMatrixes")
        plot_LC = GenerateLearningCurves(ham_data, 'Ham', tracing, model_type, "Tracing_LearningCurves")
        
        ham_data['Trial.number']

        plotsStep2_CM.append(plot_CM)
        plotsStep2_LC.append(plot_LC)

        
        #############################################################################################

        predictedData_P3_P = phishing_data[phishing_data['Training.Phase'] == 3]

        # Filter data where 'Ground Truth' is 'Phishing' 

        human_predicted_P =  np.array(predictedData_P3_P['user.choice.made.decision']).astype(str)
        model_predicted_P  = np.array(predictedData_P3_P['model.choice.made.decision']).astype(str)
        plot_CM = GenerateConfusionMatrix(human_predicted_P , model_predicted_P ,'Phishing Phase 3',tracing, model_type, "Tracing_ConfusionMatrixes")
        plot_LC = GenerateLearningCurves(predictedData_P3_P ,'Phishing Phase 3',tracing, model_type, "Tracing_LearningCurves")

        plotsStep3_CM.append(plot_CM)
        plotsStep3_LC.append(plot_LC)

        # Filter data where 'Ground Truth' is 'Ham'
        
        predictedData_P3_H = ham_data[ham_data['Training.Phase'] == 3]

        human_predicted_H =  np.array(predictedData_P3_H['user.choice.made.decision']).astype(str)
        model_predicted_H  = np.array(predictedData_P3_H['model.choice.made.decision']).astype(str)
        plot_CM = GenerateConfusionMatrix(human_predicted_H , model_predicted_H ,'Ham Phase 3',tracing, model_type, "Tracing_ConfusionMatrixes")
        plot_LC = GenerateLearningCurves(predictedData_P3_H ,'Ham Phase 3',tracing, model_type, "Tracing_LearningCurves")

        plotsStep3_CM.append(plot_CM)
        plotsStep3_LC.append(plot_LC)

        
        #############################################################################################
            
    CombinePlots(plotsStep1_CM, 1, 2, "Model-Human Syncronization", tracing, model_type, 'Human-Model', "Tracing_ConfusionMatrixes")
    CombinePlots(plotsStep1_LC, 1, 2, "Model-Human Syncronization", tracing, model_type, 'Human-Model', "Tracing_LearningCurves")
    CombinePlots(plotsStep2_CM, 2, 2, "Model-Human Syncronization split by GroundTruth", tracing, model_type, 'Split_GroundTruth', "Tracing_ConfusionMatrixes")
    CombinePlots(plotsStep2_LC, 2, 2, "Model-Human Syncronization split by GroundTruth", tracing, model_type, 'Split_GroundTruth', "Tracing_LearningCurves")
    CombinePlots(plotsStep3_CM, 2, 2, "Model-Human Syncronization Phase 3 split by GroundTruth", tracing, model_type, 'Split_GroundTruthP3', "Tracing_ConfusionMatrixes")
    CombinePlots(plotsStep3_LC, 2, 2, "Model-Human Syncronization Phase 3 split by GroundTruth", tracing, model_type, 'Split_GroundTruthP3', "Tracing_LearningCurves")

        
    sys.exit()

  