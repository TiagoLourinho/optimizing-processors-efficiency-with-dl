config = {
    "benchmarks_to_training_data": {
        ###########################
        ########## Paths ##########
        # (None values should be loaded from the .env file but kept here for info, will fail with the default values)
        ###########################
        "benchmarks_folder": None,  # str - The folder where the CUDA benchmarks are located
        "nvcc_path": None,  # str - The path of the NVCC compiler
        ################################
        ########## GPU config ##########
        ################################
        # Currently has the clock levels for sydney
        "memory_levels": {
            "max": 15001,
            "min": 7001,
            "count": 3,
        },  # dict - Defines the memory levels to use for the GPU
        "graphics_levels": {
            "max": 3090,
            "min": 1500,
            "count": 50,
        },  # dict - Defines the graphics levels to use for the GPU
        #################################
        ######## Sampling config ########
        #################################
        "sampling_freq": 100,  # int [Hz] - The sampling frequency to use (maximum is 100 Hz).
        "n_runs": 3,  # int [count] - The amount of times to run the program to get the average GPU metrics.
        "timeout_seconds": 10 * 60,  # int [seconds] - The timeout for each benchmark
    },
    "models_trainer": {
        ######################################
        ########## Training Related ##########
        ######################################
        "random_seed": 42,  # int - The seed used for random things
        "epochs": 100,  # int - Number of epochs to train
        "train_percent": 0.8,  # float - Percent of data to use to train (rest is test)
        "batch_size": None,  # int | None - The batch size (or None for no batch)
        "learning_rate": 0.0001,  # float - The optimizer learning rate
        "max_grad_norm": 1.0,  # float - The maximum gradient norm for gradient clipping
        ##########################################
        ########## Architecture Related ##########
        ##########################################
        "categorical_embedding_dim": 16,  # int - Dimension of the embeddings for categorical features
        "lstm_hidden_dim": 128,  # int - Hidden dimension of the LSTM
        "lstm_layers": 2,  # int - Number of layers in the LSTM
        "use_dynamic_info": True,  # bool - Whether to use runtime information
        "fnn_layers": 2,  # int - Number of fully connected layers
        "fnn_hidden_dim": 256,  # int - Number of hidden units in fully connected layers
        "dropout_rate": 0.0,  # float - Dropout rate for regularization
    },
}
