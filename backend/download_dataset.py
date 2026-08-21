from huggingface_hub import hf_hub_download

print("Downloading Hindi dataset...")

file_path = hf_hub_download(
    repo_id="ai4bharat/MSMARCO-XI",
    filename="train/hintrain.parquet",
    repo_type="dataset"
)

print("Download complete!")
print("File location:")
print(file_path)