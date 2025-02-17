import torch

# Load the entire model directly
model = torch.load("densenet201_food_classifier_full.pth", map_location=torch.device('cpu'))
model.eval()  # Set the model to evaluation mode

# Define a dummy input for exporting
dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 color channels, 224x224 image

# Export the model to ONNX
onnx_file_path = "densenet201_food_classifier.onnx"
torch.onnx.export(
    model,                    # Model to export
    dummy_input,              # Dummy input tensor
    onnx_file_path,           # File path to save the ONNX model
    export_params=True,       # Store trained parameter weights inside the model
    opset_version=11,         # ONNX opset version to use
    input_names=['input'],    # Specify input tensor names
    output_names=['output'],  # Specify output tensor names
    dynamic_axes={            # Enable variable batch size
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"Model has been exported to {onnx_file_path}")
