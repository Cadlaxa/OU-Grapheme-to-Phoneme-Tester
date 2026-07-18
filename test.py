import onnx

# Load the model
model = onnx.load("D:\OU-Grapheme-to-Phoneme-Tester\g2p-en-arpa-new\g2p.onnx")

print("--- Model Input Requirements ---")
for i in model.graph.input:
    # We use a list comprehension to handle the shape dimensions
    shape = [dim.dim_value for dim in i.type.tensor_type.shape.dim]
    print(f"Input Name: '{i.name}'")
    print(f"Input Shape: {shape}")
    print("-" * 30)