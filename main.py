import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import os

# Define dtype for loading books.csv
dtype_spec = {
    'ISBN': str,
    'Book-Title': str,
    'Book-Author': str,
    'Year-Of-Publication': str,
    'Publisher': str,
    'Image-URL-S': str,
    'Image-URL-M': str,
    'Image-URL-L': str
}

# Load the CSV file
books_df = pd.read_csv("books.csv", encoding='latin1', delimiter=';', on_bad_lines='skip', dtype=dtype_spec)

# Load pre-trained DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')


# Function to get BERT embedding for a single text
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():  # Disable gradients for inference
        outputs = model(**inputs)
    # Use the [CLS] token's embedding (first token)
    cls_embedding = outputs.last_hidden_state[0][0].numpy()
    return cls_embedding


# File names for output and progress
output_file = "books_with_embeddings.csv"
progress_file = "progress.txt"

# Determine starting row index
if os.path.exists(progress_file):
    with open(progress_file, "r") as pf:
        start_idx = int(pf.read().strip())
    print(f"Resuming from row {start_idx}.")
else:
    start_idx = 0
    print("Starting from the beginning.")

batch_size = 10
total_rows = len(books_df)

# If starting from 0, we need to write the header; otherwise, we append.
write_header = start_idx == 0

try:
    # Process the data in batches
    for idx in range(start_idx, total_rows, batch_size):
        # Select the batch
        batch_df = books_df.iloc[idx: idx + batch_size].copy()
        # Compute the embeddings for the "Book-Title" column
        batch_df['embedding'] = batch_df['Book-Title'].apply(lambda title: get_bert_embedding(title))
        # Convert numpy arrays to lists for CSV storage
        batch_df['embedding'] = batch_df['embedding'].apply(lambda x: x.tolist())

        # Write the current batch to the CSV file
        batch_df.to_csv(output_file, mode='a', header=write_header, index=False)
        # After the first batch, do not write header again
        write_header = False

        # Update progress (save the next starting row)
        next_idx = idx + batch_size
        with open(progress_file, "w") as pf:
            pf.write(str(next_idx))

        print(f"Processed rows {idx} to {min(next_idx, total_rows)} out of {total_rows}.")

except KeyboardInterrupt:
    print(f"Process interrupted at row {idx}. Progress saved in {progress_file}.")
