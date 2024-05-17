
# We are still working in the parts the of code, there maybe changes we can have as the status of code progresses.

#Importing necessary libraries.

from pyibl import Agent
from random import random
from datetime import datetime

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pyibl import DelayedResponse

import math
import matplotlib .pyplot as plt
import csv
import pandas as pd
import random
import numpy as np
import scipy.stats as stats
import os
import sys
import json


script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(script_dir,"Data")

# Data Loading and Preprocessing

## email_database.xlsx is the information present in PhisingDataset_HFES2019.csv from OSF, see README for more details
df = pd.read_excel(os.path.join(DATA_FILE, 'email_database.xlsx'))

#print(df)
def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)
# storing the removed tags text as list and then making it into dataframe columns.
x = []
for i in range(241):
  x.append(remove_html_tags(df['Email'][i]))
df['HTMR']= x

# to check whether the new column in the df is created or not.
df = df.replace('ham','Ham')


# Load the similarity data from CSV files
Email_SimVal = pd.read_csv(os.path.join(DATA_FILE, "similarity_matrix_emails_nli_large.csv"))
Subj_SimVal = pd.read_csv(os.path.join(DATA_FILE, "similarity_matrix_sender_nli_large.csv"))
Sender_SimVal = pd.read_csv(os.path.join(DATA_FILE, "similarity_matrix_subject_nli_large.csv"))

def limit_to_one(value):
    # Ensure the value is not less than 0
    value = max(value, 0.0)
    # Ensure the value is not greater than 1
    value = min(value, 1.0)
    return value


# Apply the limit_to_one function to all values in the DataFrames
Email_SimVal = Email_SimVal.applymap(limit_to_one)
Subj_SimVal = Subj_SimVal.applymap(limit_to_one)
Sender_SimVal = Sender_SimVal.applymap(limit_to_one)


########################


def StartTracing(experimentFile, tracing, decay, best_decayFile,fitting):


  # Initialize a dictionary to store participant data for each decay
  decay_data = []

  file_path = os.path.join(DATA_FILE, experimentFile)


  ## read the experiment file
  expData = pd.read_csv(file_path)

  ## remove all the trials where users where answering the attention check question
  expData = expData[~expData['trial'].str.contains('ch')]

  if len(best_decayFile) > 0:
    file_path = os.path.join(DATA_FILE, best_decayFile)
    best_decay = pd.read_csv(file_path)
  else:
    best_decay=[]


  # Replace values in the "email_type" column
  expData['email_type'] = expData['email_type'].replace({'ham': 'Ham', 'phishing': 'Phishing'})
  expData['user_action1'] = expData['user_action1'].replace({0: 'Ham', 1: 'Phishing'})

  ## in the dataset the user action regarind the first question
  ## user_action1	= Email is phishing?						0 - No
	## 																					1 - Yes



  expData["phase"] = pd.to_numeric(expData["phase"], errors="coerce")
  expData["trial"] = pd.to_numeric(expData["trial"], errors="coerce")

  ## order the data by "Mturk_id", "phase", "trial"
  
  column_order = ["Mturk_id", "phase", "trial"]
  # Sort the DataFrame based on the specified columns
  expData = expData.sort_values(by=column_order)

  trial = 0

  writer = []
  for p in range(0,len(expData)):
    
    userID = expData['Mturk_id'].iloc[p]
    userBaseRate = expData['base_rate'].iloc[p]
    userPhase = expData['phase'].iloc[p]
    userTrial = expData['trial'].iloc[p]
    groundTruth = expData['email_type'].iloc[p]
    userEmailID = expData['email_id'].iloc[p]
    userDecision = expData['user_action1'].iloc[p]

    if userPhase == 1:

      if userTrial == 1:

        if "75" in userBaseRate:
          base_rate = "HiFQA"
          freqBR = "High(75%)"
          
        elif "50" in userBaseRate:
          base_rate = "MeFQA"
          freqBR = "Medium(50%)"
        else:
          base_rate = "LoFQA"
          freqBR = "Low(25%)"
        
        if tracing :
          try:
            ## if we have a list with personalize decays 
            if len(best_decay) > 0:
              personalizeDecay = best_decay.loc[best_decay['UserID'] == userID]
              decay = personalizeDecay.iloc[0]['Decay_Value']

            IBL_Agent = Agent(["Decision","Email","Subject","Sender"], name=f"{base_rate}",
                      default_utility=10.0,noise = False,
                      decay = decay, mismatch_penalty=10.0, 
                      default_utility_populates = True)
          except Exception as e:
            print("An error occurred during the search of personalizeDecay:", e)
        else:## Model with default parameters
          IBL_Agent = Agent(["Decision","Email","Subject","Sender"], name=f"{base_rate}",
                      default_utility=10.0, noise = 0.25,
                      decay = 0.5, mismatch_penalty=10.0, 
                      default_utility_populates = True)
        
        IBL_Agent.reset()

        IBL_Agent.similarity(["Email"], lambda x, y: Email_SimVal.iloc[x][y])
        IBL_Agent.similarity(["Subject"], lambda x, y: Subj_SimVal.iloc[x][y])
        IBL_Agent.similarity(["Sender"], lambda x, y: Sender_SimVal.iloc[x][y])

        # choice array
        choice3 = np.array([str]*10)

        #index for positions on the above arrays
        index = 0


        outcomePrePrepoplate = np.array([0.0]*10)
        outcomeTrainPrepoplate = np.array([0.0]*40)
        outcomePost = np.array([0.0]*10)

        ## We will only consider post-training size, 
        ## since we use pre-training and training as prepopulate data
        participantData = [[] for _ in range(10)]


      if userTrial <= 10:

        outcomePrePrepoplate[index] = 1.0 if userDecision == groundTruth else 0

        IBL_Agent.populate([(userDecision,userEmailID,userEmailID,userEmailID)], outcomePrePrepoplate[index])
        
        writer = WriteDataToDictionary(IBL_Agent.name, freqBR, userID, userPhase, userTrial, 
                                            userEmailID, groundTruth, userDecision, userDecision, 
                                            outcomePrePrepoplate[index], writer, tracing, decay, '-', '-', '-', '-')
        index += 1
        # updating the outcome for pretraining after completion of 10 trials.
      
      if userTrial == 10:
        index = 0
        
    if userPhase == 2:

      ## We use all emails on this stage as prepopulate
      #outcomeTrainPrepoplate[index] = 1.0 if userDecision == df["Email_type"][userEmailID] else 0
      outcomeTrainPrepoplate[index] = 1.0 if userDecision == groundTruth else 0
      
      IBL_Agent.populate([(userDecision,userEmailID,userEmailID,userEmailID)], outcomeTrainPrepoplate[index])

      writer = WriteDataToDictionary(IBL_Agent.name, freqBR, userID, userPhase, userTrial, 
                                      userEmailID, groundTruth, userDecision, userDecision, 
                                      outcomeTrainPrepoplate[index], writer, tracing, decay, '-', '-', '-', '-')
      
      index += 1
      
      if userTrial == 40:
        index = 0
    
    ## Since there is no other phase after this it doesnt make a difference in doing the delay feedback
    if userPhase == 3:
      choicePost, d_choice = IBL_Agent.choose([('Phishing',userEmailID,userEmailID,userEmailID),('Ham', userEmailID,userEmailID,userEmailID)], True)
      choice3[index] = choicePost[0]


      for details in d_choice:
        op = details['choice']
        if op[0] == 'Phishing':
            bl_Phishing = round(details['blended_value'], 3)
            pRetrieval_Phishing = json.dumps(details['retrieval_probabilities'])
        else:
            bl_Ham = round(details['blended_value'], 3)
            pRetrieval_Ham = json.dumps(details['retrieval_probabilities'])


      if choice3[index] == userDecision or not tracing:
        outcomePost[index] = 1.0 if choice3[index] == groundTruth else 0
        IBL_Agent.respond(outcomePost[index])
      else:
        outcomePost[index] = 1.0 if userDecision == groundTruth else 0
        IBL_Agent.respond(outcomePost[index], (userDecision,userEmailID,userEmailID,userEmailID))
      
      writer = WriteDataToDictionary(IBL_Agent.name, freqBR, userID, userPhase, userTrial, userEmailID, groundTruth, choice3[index], 
                                      userDecision, outcomePost[index], writer, tracing, decay, bl_Phishing, bl_Ham, 
                                      pRetrieval_Phishing, pRetrieval_Ham)

      participantData [index] = [userID, decay, int(userPhase), userTrial, 
                                    1.0 if userDecision == "Phishing" else 0, 1.0 if choice3[index] == "Phishing" else 0, 
                                    groundTruth, 1.0 if userDecision == choice3[index] else 0]

      index += 1
      
      if userTrial == 10:
        decay_data.append(participantData)
    
    
  # Convert the array to a DataFrame
  df_to_append = pd.DataFrame(writer)  

   # Specify the CSV file name
  if fitting:
    csv_file_name = f'Tracing_Data_Fitting_M2.csv'
  elif tracing and len(best_decay) == 0:
    csv_file_name = f'Tracing_Data_decay_{IBL_Agent.decay}_M2.csv'
  elif tracing and len(best_decay) > 0:
     csv_file_name = f'Tracing_Data_Personalized_M2.csv'
  else:
    csv_file_name = 'IBL_Data_M2.csv'

  # Specify the desired header
  desired_header = ['Agent','Frequency(Base_Rate)','Mturk_id','Training.Phase','Trial.number','Email_ID','ground.truth', 'model.choice.made.decision', 
                    'user.choice.made.decision','modelSyncUser', 'outcome','Hit','Miss','False','Correct.Rejection', 'Tracing', 'Decay_Value', 
                    'Blended_Value_Phishing', 'Blended_Value_Ham', 'Retrieval_Probabilities_Phishing', 'Retrieval_Probabilities_Ham']

  
  # Check if the CSV file already exists
  try:
    # Read the existing CSV file
    if fitting:
      file_path = os.path.join(script_dir, "Generated_Models_Data", "Tracing_Results_Fitting_M2")
    elif tracing:
      file_path = os.path.join(script_dir, "Generated_Models_Data", "Tracing_Results_M2")
    else:
      file_path = os.path.join(script_dir, "Generated_Models_Data", "IBL_Results_M2")
    os.makedirs(file_path, exist_ok=True)
    existing_df = pd.read_csv(os.path.join(file_path, csv_file_name))

    # Append the new data to the existing DataFrame
    updated_df = df_to_append

  except FileNotFoundError:
      # If the file doesn't exist, create a new DataFrame
      updated_df = df_to_append

# Check if the header needs to be added
  if not os.path.isfile(os.path.join(file_path, csv_file_name)):
      # If the file doesn't have a header, add it
      updated_df.to_csv(os.path.join(file_path, csv_file_name), mode='w', index=False, header=desired_header)
  else:
      # If the file already has a header, append without writing the header again
      updated_df.to_csv(os.path.join(file_path, csv_file_name), mode='a', index=False, header=False)


  ######## Run the line below only when doing model-fitting
  #calculateFittingMetric(decay_data, tracing, fitting)


def WriteDataToDictionary(AgentName,freqBR, userID, userPhase, userTrial, userEmailID, groundTruth, choiceModelMade, 
                          userDecision, outcome, writer, tracing, decay, bl_Phishing, bl_Ham,
                          pRetrieval_Phishing, pRetrieval_Ham):


  writer.append({'Agent': AgentName,
                'Frequency(Base_Rate)': freqBR,
                'Mturk_id': userID,
                'Training.Phase':userPhase,
                'Trial.number': userTrial,
                'Email_ID':userEmailID,
                'ground.truth': groundTruth, 
                'model.choice.made.decision': choiceModelMade, 
                'user.choice.made.decision': userDecision,
                'modelSyncUser': 1 if choiceModelMade == userDecision else 0,
                'outcome': outcome,
                'Hit':1 if choiceModelMade == "Phishing" and groundTruth == "Phishing" else 0,
                'Miss':1 if choiceModelMade == "Ham" and groundTruth == "Phishing" else 0,
                'False':1 if choiceModelMade == "Phishing" and groundTruth == "Ham" else 0,
                'Correct.Rejection':1 if choiceModelMade == "Ham" and groundTruth == "Ham" else 0,
                'Tracing':tracing,
                'decay_value':decay,
                'bl_Phishing':bl_Phishing, 
                'bl_Ham':bl_Ham,
                'probRetrieval_Phishing':pRetrieval_Phishing, 
                'probRetrieval_Ham':pRetrieval_Ham})
  
  
  return writer


def calculateFittingMetric(decay_data, tracing):
  results = []
  # Initialize lists to store user decisions and model decisions
 
  try:
    for data in decay_data:

      # Extract decay value
      userID = data[0][0]
      decay_value = data[0][1]  # Assuming decay is constant for each file
      
       # Extract the relevant columns based on the array bellow
      #participantData [trial-1] = [userID, d_value, userPhase, userTrial, userDecision, choice3[index], groundTruth, syncRate]

      syncRate_coded = [row[7] for row in data]

      SyncRate = sum(syncRate_coded)/len(syncRate_coded)


      results.append([userID, decay_value, SyncRate, tracing])

    #### Saving the data into a file ####
    
    # Specify the desired header
    desired_header = ['UserID','Decay_Value','SyncRate_Value', 'Tracing']

    # Check if the CSV file already exists
    csv_file_name = 'MaxSyncRate_Fitting_Data_M2.csv'
    # Convert the array to a DataFrame
    df_to_append = pd.DataFrame(results) 
    checkIfFileExists(csv_file_name, df_to_append, desired_header)
  except Exception as e:
    print(f"Error retrieving calculateFittingMetric: {e}")
    

def checkIfFileExists(csv_file_name, df_to_append, desired_header):
  try:
    # Read the existing CSV file
    file_path = os.path.join(script_dir, "Generated_Models_Data")
    os.makedirs(file_path, exist_ok=True)
    existing_df = pd.read_csv(os.path.join(file_path, csv_file_name))

    # Append the new data to the existing DataFrame
    updated_df = df_to_append

  except FileNotFoundError:
    # If the file doesn't exist, create a new DataFrame
    updated_df = df_to_append
  
  # Check if the header needs to be added
  if not os.path.isfile(os.path.join(file_path, csv_file_name)):
    # If the file doesn't have a header, add it
    updated_df.to_csv(os.path.join(file_path, csv_file_name), mode='w', index=False, header=desired_header)
  else:
    # If the file already has a header, append without writing the header again
    updated_df.to_csv(os.path.join(file_path, csv_file_name), mode='a', index=False, header=False)


if __name__ == "__main__":


  experimentFile = "experiment1-outcomefeedback.csv"
  
  best_decay = "max_decays_M2.csv"
  
  ## Runs model-tracing with personalized decay per participant
  StartTracing(experimentFile, tracing=True, decay=0, best_decayFile=best_decay, fitting=False)

  ## Runs model-tracing with default decay per participant
  #StartTracing(experimentFile, tracing=True, decay=0.5, best_decayFile=[], fitting=False)

  ## Runs model without tracing (making its own decisions) with default decay per participant
  #StartTracing(experimentFile, tracing=False, decay=0.5, best_decayFile=[], fitting=False)




  ## Uncomment to run model-fitting with model-tracing, ideally comment the StartTracing above
  """ decay_data = []

  start = 0.10
  end = 3.00
  step = 0.01
  d_values = [round(start + i * step, 2) for i in range(int((end - start) / step) + 1)]

  try:
    for d_value in d_values:
      StartTracing(experimentFile, tracing=True, decay=d_value, best_decayFile=[], fitting=True)
  except Exception as e:
    print(f"Error retrieving StartTracing for each decay: {e}") """

  sys.exit()