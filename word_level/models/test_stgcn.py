import torch
from stgcn_model import STGCNModel


print("=" * 60)
print("ST-GCN MODEL TEST")
print("=" * 60)


# Create the model
model = STGCNModel(
    input_channels=3,
    hidden_channels=64,
    num_classes=10
)

print("\nModel created successfully!")


# Create a fake input
# Batch = 1
# Frames = 30
# Landmarks = 21
# Coordinates = 3

x = torch.randn(
    1,
    30,
    21,
    3
)

print("Input shape:", x.shape)


# Run the model
with torch.no_grad():

    output = model(x)


print("Output shape:", output.shape)

print("\nExpected output shape:")
print("(1, 10)")


# Verify the shape
if output.shape == (1, 10):

    print("\nSUCCESS: ST-GCN model is working correctly!")

else:

    print("\nERROR: Unexpected output shape.")