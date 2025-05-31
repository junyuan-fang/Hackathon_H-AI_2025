# from huggingface_hub import snapshot_download

# # local_dir = snapshot_download("sy1998/Video_XL")
# local_dir = snapshot_download("openai/clip-vit-large-patch14-336")
# print(f"Downloaded to {local_dir}") 

from transformers import CLIPProcessor, CLIPModel

# Load the model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

# Save the model and processor locally
model.save_pretrained("./clip-vit-large-patch14-336")
processor.save_pretrained("./clip-vit-large-patch14-336")

