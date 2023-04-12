# CS6910 Assignment 2

## Part B
The file code utilizes PyTorch and the PyTorch Lightning framework to train a fine-tuned **ResNet18 model**  on a custom dataset. The model will be trained to classify images into different classes
### Instructions to train and evaluate the model
1. Install the required libraries:
```python
!pip install pytorch_lightning
!curl -SL https://storage.googleapis.com/wandb_datasets/nature_12K.zip > Asg2_Dataset.zip
!unzip Asg2_Dataset.zip
!pip install wandb
```
2. Give proper path for the dataset.
3. To fine-tune pre-trained model, use the following command
```python
net = FineTuneTask(10)
trainer = pl.Trainer(max_epochs=5, accelerator="gpu", devices=1)
trainer.fit(model=net,train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)
```
4. To evaluate the model, use the following command:
```python
trainer.test(net,test_dataloader)
```
### Dataset and Data Loaders
The iNaturalist 12K dataset is loaded from the \
/kaggle/input/inaturalist12k/Data/inaturalist_12K/train  and \
/kaggle/input/inaturalist12k/Data/inaturalist_12K/val \
directories for training and testing (if we are using kaggle), respectively. Depending on the value of the data_augmentation parameter in the project configuration, either transform or transform_augmented is applied to the training set. The testing set always uses transform.

The dataset is split into training and validation sets using a ratio of 80:20. Data loader objects are created for both sets, with a batch size of 64 for both and the training set being shuffled.
    
### Data Transformations
Two sets of data transformations are defined: transform and transform_augmented. Both transform the images to have a size of 256x256 pixels and convert them to tensors. However, transform_augmented applies additional data augmentation techniques to the images, including random cropping, flipping, and rotating. Both transformations then normalize the images using the mean and standard deviation values for the ImageNet dataset.

### Methods
The __init__ method creates a new ResNet18 model with a custom output layer that has num_classes number of neurons. The forward method defines how an input tensor is passed through the model.

Training step method defines what happens during a single training step. It calculates the loss and accuracy of the model on a batch of training data and appends the values to the appropriate lists.

Similarly, the validation_step method calculates the loss and accuracy of the model during validation and logs them. 
Finally, the test_step method calculates the loss and accuracy of the model during testing.



