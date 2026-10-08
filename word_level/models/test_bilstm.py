import torch
from bilstm_model import BiLSTMModel


# Create the model
model = BiLSTMModel()

print("=" * 50)
print("BiLSTM MODEL TEST")
print("=" * 50)

print("Model created successfully!")

# Create one fake sequence
sample = torch.randn(1, 30, 21, 3)

print("Input shape:", sample.shape)

# Send the sample through the model
output = model(sample)

print("Output shape:", output.shape)

print("\nExpected output shape: (1, 10)")

if output.shape == (1, 10):
    print("\nSUCCESS: BiLSTM model is working correctly!")
else:
    print("\nERROR: Unexpected output shape.")