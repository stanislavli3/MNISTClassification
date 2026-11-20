# MNIST Classification with PyTorch

A compact PyTorch notebook that trains a fully connected neural network to recognize handwritten digits (0 to 9) from the MNIST dataset.

Built for **AI for Cyber-Physical Systems (AI4CPS)** at Lehigh University.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stanislavli3/MNISTClassification/blob/main/AI4CPS%3A%20MNIST%20Classification%20using%20PyTorch.ipynb)

## The model

`MyNet` is a small multilayer perceptron:

| Layer | Shape | Activation |
|---|---|---|
| Input | 28 x 28 image flattened to 784 | |
| `fc1` (Linear) | 784 to 128 | ReLU |
| `fc2` (Linear) | 128 to 10 | logits (softmax applied inside the loss) |

That is 101,770 trainable parameters, small enough to train comfortably on a CPU.

## Training setup

| Setting | Value |
|---|---|
| Data | MNIST training set (60,000 images), downloaded with `torchvision.datasets.MNIST` |
| Preprocessing | `ToTensor()` then `Normalize(mean=0.1307, std=0.3081)` |
| Split | 80 / 10 / 10 with `random_split`: 48,000 train, 6,000 validation, 6,000 held out |
| Loss | Cross entropy |
| Optimizer | Adam, learning rate 0.001 |
| Batch size | 64 (train and validation), 1,000 (held out) |
| Epochs | 10 |
| Device | CUDA if available, otherwise CPU |

The training loop follows the standard PyTorch pattern (move batch to device, zero gradients, forward pass, loss, backward pass, optimizer step) and prints the mean training loss after each epoch.

## Results

The committed notebook does not include saved outputs, so no accuracy or loss numbers are recorded here. Running it prints the average training loss for each of the 10 epochs.

To measure accuracy on the held out split, add a cell like this after training:

```python
model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
print(f"Held out accuracy: {correct / total:.2%}")
```

The same loop over `val_loader` gives validation accuracy for tuning.

## How to run

**Google Colab (easiest):** click the badge above. If Colab asks about the runtime, choose **Runtime > Change runtime type > Python 3** (the notebook metadata names a different kernel). PyTorch and torchvision are preinstalled. Run the cells top to bottom.

**Local Jupyter:**

```bash
git clone https://github.com/stanislavli3/MNISTClassification.git
cd MNISTClassification
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision numpy jupyter
jupyter notebook
```

Open the notebook, select a Python 3 kernel, and run all cells. MNIST downloads to `./data` on the first run.

## What I learned

- **Input normalization matters.** Scaling pixels with the dataset mean and standard deviation keeps inputs centered and on a consistent scale, which makes optimization better behaved.
- **Splitting data before you train.** Carving validation and held out sets out of the training data with `random_split` keeps evaluation honest and leaves room for tuning.
- **The core training loop.** Writing the forward, loss, backward and step sequence by hand makes it clear what higher level trainers automate.
- **Logits and loss pairing.** `CrossEntropyLoss` expects raw logits, so the network ends in a plain linear layer with no softmax.
- **Device placement.** Model and every batch must live on the same device, which is what lets the same code run on CPU or GPU.
