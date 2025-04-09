config = {
    "benchmarks_to_training_data": {
        ###########################
        ########## Paths ##########
        ###########################
        "benchmarks_folder": "benchmarks",  # str - The folder where the CUDA benchmarks are located
        "nvcc_path": "/usr/local/cuda-12.6/bin/nvcc",  # str - The path of the NVCC compiler
        "ncu_path": "/usr/local/cuda-12.6/bin/ncu",  # str - The path of the NCU profiler
        "ncu_sections_folder": None,  # str - The path of the folder to search for NCU sections. If None, then the default path is used.
        "ncu_python_report_folder": "/opt/nvidia/nsight-compute/2024.3.2/extras/python/",  # str - The path of the folder where the NCU python report interface is located
        ################################
        ########## GPU config ##########
        ################################
        "n_core_clocks": 20,  # int - The amount of core clocks to consider when collecting samples (GPUs have like >120 which might be too many)
        "gpu_sleep_time": 0.5,  # float [s] - The amount of time to sleep after changing GPU clocks speeds so the systems stabilizes.
        #################################
        ########## NVML config ##########
        #################################
        "nvml_sampling_freq": 100,  # int [Hz] - The NVML sampling frequency to use (maximum is 100 Hz).
        "nvml_n_runs": 3,  # int [count] - The amount of times to run the program to get the median GPU metrics.
        ################################
        ########## NCU config ##########
        ################################
        "ncu_set": "basic",  # str - The name of the set of metrics to collect using NCU.}
    },
    "models_trainer": {
        ######################################
        ########## Training Related ##########
        ######################################
        "random_seed": 1,  # int - The seed used for random things
        "epochs": 20,  # int - Number of epochs to train
        "train_percent": 0.9,  # float - Percent of data to use to train (rest is test)
        "batch_size": None,  # int | None - The batch size (or None for no batch)
        "learning_rate": 0.001,  # float - The optimizer learning rate
        "runtime_loss_weight": 1,  # float - The weight for the runtime loss when combining losses
        "power_loss_weight": 1,  # float - The weight for the power loss when combining losses
        ##########################################
        ########## Architecture Related ##########
        ##########################################
        "categorical_embedding_dim": 16,  # int - Dimension of the embeddings for categorical features
        "lstm_hidden_dim": 128,  # int - Hidden dimension of the LSTM
        "lstm_layers": 1,  # int - Number of layers in the LSTM
        "fnn_hidden_dim": 128,  # int - Number of hidden units in fully connected layers
        "dropout_rate": 0.3,  # float - Dropout rate for regularization
    },
}
