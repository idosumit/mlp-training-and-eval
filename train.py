import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from mlp import MLP

# creating a small dataset
X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])

y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])

y_test = torch.tensor([0, 1])

# creating a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        indiv_x = self.features[idx]
        indiv_y = self.labels[idx]
        return indiv_x, indiv_y

train_ds = CustomDataset(X_train, y_train)
test_ds = CustomDataset(X_test, y_test)

# checking the length of the datasets
print(f"Number of samples in train_ds: {len(train_ds)}")
print(f"Number of samples in test_ds: {len(test_ds)}")

# instantiating data loaders
torch.manual_seed(123456578)

train_loader = DataLoader(
    train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    drop_last=False
)

# iterating over the train_loader
print("\n======== Iterating Over the Train Loader ========")
for idx, (X, y) in enumerate(train_loader):
    print(f"\nBatch {idx+1}:\nx: {X}\ny: {y}\n")

# creating the nn model and training
torch.manual_seed(12345678)

model = MLP(2, 2) # 2 inputs, 2 outputs

optimizer = torch.optim.SGD(
    params = model.parameters(),
    lr = 0.5
)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()

    for batch_idx, (features_aka_X, labels_aka_y) in enumerate(train_loader):
        logits = model(features_aka_X)
        loss = F.cross_entropy(logits, labels_aka_y)

        optimizer.zero_grad() # setting previous grad to 0
        loss.backward() # backprop

        optimizer.step() # upating model parameters using gradients

        print(f'''Epoch: {epoch+1:03d}/{num_epochs:03d}
            Batch {batch_idx+1:03d}/{len(train_loader):04d}
            Train Loss: {loss:.2f}
            ''')

    model.eval()
        # optional model eval code here

# model training is done. doing the output:
model.eval()
with torch.no_grad():
    outputs = model(X_train)

# softmax to obtain class membership probabilities
'''this is optional since this is a very simple MLP, so just argmax would suffice. but doing it for practice.
'''
torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)
print(f"\nClass Membership Probabilities:\n{probas}")

# creating predictions based on probabilities
predictions = torch.argmax(probas, dim=1)
print(f"\nPredictions: {predictions}")

print("\nprediction == y_train gives us:\n", predictions == y_train)

print(f"Total amount of times the prediction is true: {torch.sum(predictions == y_train)}\n")

# one final thing: a function to compute the prediction accuracy in a nicer way
def compute_accuracy(model, dataloader):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)

        # not doing softmax since this model doesn't need it (redundancy reasons)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (f"{(correct / total_examples).item() * 100}% accuracy")

print(f"Accuracy of train_loader: {compute_accuracy(model, train_loader)}\nAccuracy of test_loader: {compute_accuracy(model, test_loader)}")
