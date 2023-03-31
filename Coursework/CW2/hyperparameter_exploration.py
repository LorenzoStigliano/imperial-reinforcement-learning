from utils import train_agent
from itertools import product

import numpy as np
import optuna
import torch
import torch.nn.functional as F

def objective_function(runs_results):
    results = torch.tensor(runs_results)
    means = results.float().mean(0)
    return (len(means.detach().numpy()) - np.count_nonzero(means.detach().numpy()>100))

def objective(trial):
    
    #exploration vs exploitation
    EPSILON = trial.suggest_uniform("epsilon", 0, 1)  # epislon - action to take     
    #EPSILON_DECAY = trial.suggest_categorical('epsilon_decay', [0.9, 0.925, 0.95, 0.975, 1]) #if epsilon decay is 1 then there is no decay
    EPSILON_DECAY = 0.99

    #optimiser
    OPTIMISER_TYPE = trial.suggest_categorical('optimiser', ['Adam', 'RAdam', 'SGD'])
    LEARNING_RATE = trial.suggest_loguniform("lr", 1e-5, 1)
    #WEIGHT_DECAY = trial.suggest_loguniform("lr", 0, 0.1)
    WEIGHT_DECAY = 0

    #neural network 
    POLICY_NN = trial.suggest_categorical('policy_network_nn', [ 
        [4, 32, 2], 
        [4, 64, 2], 
        [4, 32, 32, 2],
        [4, 64, 32, 2], 
        [4, 64, 64, 2],
        [4, 64, 128, 2],  
        [4, 64, 64, 2],  
        [4, 128, 128, 2], 
        [4, 32, 32, 32, 2], 
        [4, 64, 64, 64, 2], 
        [4, 128, 128, 128, 2], 
    ])
    ACTIVATION_FUNCTION = trial.suggest_categorical('activation_function',[F.relu,  F.leaky_relu, F.sigmoid, F.tanh])

    #replay
    BATCH_SIZE = [1, 2, 5, 10, 20]
    MEMORY_REPLAY_BUFFER_SIZE = [500, 1000, 2000, 5000, 10000, 20000]   
    BATCH_SIZE_BUFFER_SIZE_ARRAY = [i for i in list(product(BATCH_SIZE, MEMORY_REPLAY_BUFFER_SIZE)) if i[0] <= i[1]]
    BATCH_SIZE_BUFFER_SIZE = trial.suggest_categorical('batch_size_buffer_size', BATCH_SIZE_BUFFER_SIZE_ARRAY)

    #update target network
    ITERATIONS_TO_UPDATE_TARGET_NN = trial.suggest_categorical('update_target_policy_every_n', [1, 2, 5, 10, 20, 40])
    
    runs_results = train_agent(
                        EPSILON, 
                        EPSILON_DECAY,
                        LEARNING_RATE, 
                        WEIGHT_DECAY,
                        BATCH_SIZE_BUFFER_SIZE,
                        POLICY_NN, 
                        ITERATIONS_TO_UPDATE_TARGET_NN,
                        ACTIVATION_FUNCTION, 
                        OPTIMISER_TYPE
                    )
    
    value = objective_function(runs_results)

    if value < 0:
        good_parameters = [EPSILON, 
                    EPSILON_DECAY,
                    LEARNING_RATE, 
                    WEIGHT_DECAY,
                    BATCH_SIZE, 
                    MEMORY_REPLAY_BUFFER_SIZE,
                    POLICY_NN,                         
                    ITERATIONS_TO_UPDATE_TARGET_NN,
                    ACTIVATION_FUNCTION, 
                    OPTIMISER_TYPE]
        print(value, good_parameters)
    
    return value

def create_optuna_study():
    study = optuna.create_study(study_name="cartpole")
    study.optimize(objective, n_trials=30)
    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
