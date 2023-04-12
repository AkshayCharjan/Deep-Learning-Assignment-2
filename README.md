# CS6910 Assignment 2

## Part A

This code file implements a convolutional neural network (CNN) using PyTorch's Lightning library. It contains code for training a deep learning model for image classification using the iNaturalist 12K dataset. The CNN architecture consists of 5 convolutional layers followed by max pooling, batch normalization, dropout, and a fully connected layer with softmax activation. The hyperparameters of the model such as activation function, batch normalization, data augmentation, filter organization, and dropout rate can be specified as command line arguments.\

### Instructions to train and evaluate the model
1. Install the required libraries:
```python
!pip install pytorch_lightning
!curl -SL https://storage.googleapis.com/wandb_datasets/nature_12K.zip > Asg2_Dataset.zip
!unzip Asg2_Dataset.zip
!pip install wandb
```

2. To train the model run train.py using the below command: 
```python
python train.py --wandb_entity myname --wandb_project myprojectname
```
Following are the supported command line arguments:

|           Name           | Default Value | Description                                                               |
| :----------------------: | :-----------: | :------------------------------------------------------------------------ |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard      |
|  `-we`, `--wandb_entity` |     myname    | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
|  `-e`, `--epochs` |     10    | Number of epochs to train neural network.. |
|  `-lr`, `--learning_rate` |     0.0001    | Learning rate used to optimize model parameters. |
|  `-fz`, `--filter_size` |    64     | Filter size used by convolution layer. |
|  `-fo`, `--filter_organisation` |    same     | choices= [same,half,double] Filter organisation used by convolution layer. Using same would give the same filter size to all the layers, using half would keep halving the filter size in the successive layers and using double would mean doubling the filter size in the successive layers.|
|  `-da`, `--DataAugmentation` |     False    | choices = [Yes, No]. Perform data augmentation or not |
|  `-bn`, `--batch_normalisation` |     No    | choices = [Yes, No]. Perform batchNormalisation or not |
|  `-do`, `--drop_out` |     0.3    | Dropout value. |
|  `-a`, `--activation_function` |     GELU    | choices = [RELU, GELU, SELU, MISH]. Activation function to use |
\
3. <>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>TESTING INSTRUCIONS GO HERE
### Dataset and Data Loaders
The iNaturalist 12K dataset is loaded from the /kaggle/input/inaturalist12k/Data/inaturalist_12K/train and /kaggle/input/inaturalist12k/Data/inaturalist_12K/val directories for training and testing, respectively. Depending on the value of the data_augmentation parameter in the project configuration, either transform or transform_augmented is applied to the training set. The testing set always uses transform.

The dataset is split into training and validation sets using a ratio of 80:20. Data loader objects are created for both sets, with a batch size of 64 for both and the training set being shuffled.
    
### Data Transformations
Two sets of data transformations are defined: transform and transform_augmented. Both transform the images to have a size of 256x256 pixels and convert them to tensors. However, transform_augmented applies additional data augmentation techniques to the images, including random cropping, flipping, and rotating. Both transformations then normalize the images using the mean and standard deviation values for the ImageNet dataset.
    
### Methods
The training_step method calculates the loss and accuracy of the model during training and logs them using wandb_logger. Similarly, the validation_step method calculates the loss and accuracy of the model during validation and logs them. Finally, the test_step method calculates the loss and accuracy of the model during testing.

The configure_optimizers method initializes the Adam optimizer with a specified learning rate.

The main code initializes an instance of the CNNModel class with the specified hyperparameters and trains the model using the fit method of Trainer class provided by PyTorch Lightning. The wandb_logger is used to log the training and validation metrics to the Weights & Biases platform. The max_epochs and devices can also be specified as command line arguments.

### sweep
The following code sets up a configuration for a parameter sweep using the wandb.sweep function. The sweep configuration is defined as a dictionary sweep_config.
```python
sweep_config = {
    'method': 'bayes', # or 'grid'
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'drop_out': {"values": [0.2, 0.3]},
        "activation_function": {
              "values": [ "ReLU", "SiLU", "GELU", "Mish"]
          },
          
          "learning_rate": {
              "values": [1e-3, 1e-4]
          }
          ,
        "filter_organisation":{
            "values":[[8,8,8,8,8],[16,16,16,16,16],[32,32,32,32,32],[64,64,64,64,64]]
        },
        "data_augmentation":{
            "values":["Yes","No"]
        },
        "batch_normalization":{
            "values":["Yes","No"]
        },
          "epochs": {
              "values": [5, 10]
          },
    }
}
```

The method key specifies the method used for the sweep. In this case, it is set to 'bayes', which suggests that the sweep will use Bayesian optimization to determine the best set of hyperparameters.

The metric key is a dictionary that defines the metric to optimize. In this case, the metric is the validation loss, and the goal is to minimize it.

The parameters key is a dictionary that defines the hyperparameters to sweep over. The hyperparameters include:
**drop_out**: the dropout rate\
**activation_function**: the activation function to use in the model\
**learning_rate**: the learning rate for the optimizer\
**filter_organisation**: the number of filters for each convolutional layer in the model\
**data_augmentation**: whether to use data augmentation or not\
**batch_normalization**: whether to use batch normalization or not\
**epochs**: the number of epochs to train the model for\

For each hyperparameter, a list of possible values to sweep over is provided using the values key. The sweep will test different combinations of hyperparameters and record the results in Weights and Biases (W&B) for analysis.

##### Use the wandb.sweep function to set up the hyperparameter sweep
sweep_id = wandb.sweep(sweep_config, project=pName)

##### Use the wandb.agent function to run the hyperparameter sweep with the given sweep_id and sweep function
wandb.agent(sweep_id, sweep)



