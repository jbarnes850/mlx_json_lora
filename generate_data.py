from datasets import load_dataset
import json
import random

print("Loading dataset and tokenizer...")
qa_dataset = load_dataset("squad_v2")

def create_completion(context, question, answer):
    if len(answer["text"]) < 1:
        answer_text = "I Don't Know"
    else:
        answer_text = answer["text"][0]
    
    completion_template = {
        "context": context,
        "question": question,
        "answer": answer_text
    }
    
    return json.dumps(completion_template)

def process_dataset(dataset):
    processed_data = []
    for sample in dataset:
        completion = create_completion(sample['context'], sample['question'], sample['answers'])
        prompt = sample['question']
        processed_data.append({"prompt": prompt, "completion": completion})
    return processed_data

print("Processing training data...")
train_data = process_dataset(qa_dataset['train'])
print("Processing validation data...")
valid_data = process_dataset(qa_dataset['validation'])  # SQuAD v2 uses 'validation' as test set

# Combine all data for redistribution
all_data = train_data + valid_data
random.shuffle(all_data)

# Calculate new split sizes
total_size = len(all_data)
train_size = int(0.8 * total_size)
test_size = int(0.1 * total_size)
valid_size = total_size - train_size - test_size

# Split the data
new_train_data = all_data[:train_size]
new_test_data = all_data[train_size:train_size+test_size]
new_valid_data = all_data[train_size+test_size:]

# Write to JSONL files
def write_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

print("Writing train.jsonl...")
folder_prefix = "./data/"
write_jsonl(new_train_data, folder_prefix+'train.jsonl')
print("Writing test.jsonl...")
write_jsonl(new_test_data, folder_prefix+'test.jsonl')
print("Writing valid.jsonl...")
write_jsonl(new_valid_data, folder_prefix+'valid.jsonl')

print(f"Dataset split and saved: train ({len(new_train_data)}), test ({len(new_test_data)}), valid ({len(new_valid_data)})")

# Verify file contents
def count_lines(filename):
    with open(folder_prefix+filename, 'r') as f:
        return sum(1 for _ in f)

print("\nVerifying file contents:")
print(f"train.jsonl: {count_lines('train.jsonl')} lines")
print(f"test.jsonl: {count_lines('test.jsonl')} lines")
print(f"valid.jsonl: {count_lines('valid.jsonl')} lines")