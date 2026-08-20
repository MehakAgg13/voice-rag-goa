from datasets import load_dataset

print("Downloading dataset...")

# Download MSMARCO-XI
dataset = load_dataset("ai4bharat/MSMARCO-XI")

print(dataset)

# Save locally
dataset.save_to_disk("data/msmarco_xi")

print("Dataset downloaded successfully!")