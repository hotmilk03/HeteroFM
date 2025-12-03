import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    A simple 2-layer Multi-Layer Perceptron for MNIST classification.
    The size of the hidden layer is configurable.
    """
    def __init__(self, hidden_size=256):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 10)
        )

    def forward(self, x):
        """
        Forward pass of the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10).
        """
        return self.layers(x)

def init_model(hidden_size):
    """
    Initializes and returns the MLP model with a specific hidden size.
    """
    return MLP(hidden_size)

if __name__ == '__main__':
    # A quick test to verify the model's architecture
    test_hidden_size = 128
    model = init_model(hidden_size=test_hidden_size)
    print(f"Model initialized with hidden_size = {test_hidden_size}")
    print(model)
    
    # Check input/output dimensions
    try:
        dummy_input = torch.randn(64, 1, 28, 28) # Batch of 64 images
        output = model(dummy_input)
        print(f"\nInput shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        assert output.shape == (64, 10)
        print("\nModel architecture and I/O dimensions are correct.")
    except Exception as e:
        print(f"\nAn error occurred during model verification: {e}")