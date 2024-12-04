# MLP Training and Eval

This is a simple MLP with 2 hidden layers.

### MLP Structure

This is how the hidden layers are structured:

```python
# hidden layer 1
nn.Linear(num_inputs, 30),
nn.ReLU(),

# hidden layer 2
nn.Linear(30, 20),
nn.ReLU(),

# output layer
nn.Linear(20, num_outputs)
```

### Creating Dataset and DataLoader

I create a custom dataset (using PyTorch's `Dataset`) that takes in 2 tensors: `X` and `y`, which becomes useful to create datasets for both training and testing.

I also use PyTorch's `DataLoader` to load the datasets with specific `batch_size` and other individual configurations (you can see the details in the [mlp.py](./mlp.py) file. This can also be used for instantiating both train and test data loaders.

### Some other things we can play around with
- changing the provided dataset (`X_train`, `y_train`, `X_test`, `y_test`)
- changing the `batch_size` while instantiating data loaders
- changing `num_inputs` and `num_outputs` to feed to the MLP for creating our neural network model
- playing around with the optimizer
    - SGD is currently used, maybe can play around with other things
    - changing the learning rate
- changing the number of epochs for training maybe

### If we want to check the model parameters

We can run the following code after creating the model with the number of inputs + outputs we desire to get the number of parameters:

```python
model = MLP(num_inputs = 2, num_outputs = 2) # say, a created model

print(f"Total parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

```

### For checking the samples in the train_ds and test_ds for better intuition behind PyTorch's `Dataset`

```python
# Check the length of the datasets
print(f"\nNumber of samples in train_ds: {len(train_ds)}")
print(f"\nNumber of samples in test_ds: {len(test_ds)}")

# looping through the samples to inspect them
for i in range(len(train_ds)):
    print(f"\nSample {i} in train_ds:", train_ds[i])

for i in range(len(test_ds)):
    print(f"\nSample {i} in test_ds:", test_ds[i])
```
