from openai import OpenAI
import torch
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re
from transformers import RobertaTokenizer
import string
client = OpenAI(api_key="",
                    base_url=""
)



# load_path = "L:\\base_2500.pt" # path to saved tonkenized data

load_path = r"G:\JupOtter\dataset\tokenized_content\je.pt"
tokenized_data = torch.load(load_path)

# train_ids = tokenized_data['train_ids']
# test_ids = tokenized_data['test_ids']
# train_masks = tokenized_data['train_masks']
# test_masks = tokenized_data['test_masks']
# train_labels = tokenized_data['train_labels']
# test_labels = tokenized_data['test_labels']

# Uncomment the lines below to load unsplit data
test_ids = tokenized_data['test_ids']
test_masks = tokenized_data['test_masks']
test_labels = tokenized_data['test_labels']

print("Tokenized data loaded successfully.")


tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

# setting up the special tokens use for finding cell boundaries in tokenized content
start_special_tokens = [f"<CELL_{i}>" for i in range(1, 1024)]
end_special_tokens = [f"<END_CELL_{i}>" for i in range(1, 1024)]
all_special_tokens = start_special_tokens + end_special_tokens


for token in all_special_tokens:
    if token not in tokenizer.get_vocab():
        tokenizer.add_tokens([token])




flat_codes, flat_labels = [], []
i = 1
for chunks_ids, chunks_masks, chunk_label_lists in tqdm(
    zip(test_ids[:2], test_masks[:2], test_labels[:2]),
    total=len(test_ids),
    desc="Decoding & cleaning notebooks",
    dynamic_ncols=True,
):
    file_ids = chunks_ids[:4]
    is_buggy = int(any((lbls == 1).any().item() for lbls in chunk_label_lists[:4]))
    flat_list = file_ids.reshape(-1).tolist()
    decoded = tokenizer.decode(flat_list, skip_special_tokens=True)

    for token in tokenizer.all_special_tokens:
        pattern = re.escape(token)
        decoded = re.sub(pattern, "", decoded)

    decoded = re.sub(r"<CELL_\d+>", "", decoded)
    decoded = re.sub(r"<END_CELL_\d+>", "", decoded)
   
    flat_codes.append(decoded)
    flat_labels.append(is_buggy)

    # break




results = [] # holds the results of the predictions, each element is a tuple for its notebook (is_buggy, label)
correctnessOfPredictions = []

tq = tqdm(
    enumerate(zip(flat_codes, flat_labels)),
    total=len(flat_codes),
    desc="Static analysis Eval",
    dynamic_ncols=True,
    leave=True,
)
buggy_pred = 0
non_buggy_pred = 0
skipped = 0
for i, (code, label) in tq:

    messages = [
        {"role": "user", "content": f"You are doing file level bug detection. Respond YES if there is a bug in the file else NO. Only ever respond with YES or NO. {code}"},
    ]

    try:
        # create a chat completion request
        response = client.chat.completions.create(
            model="gemini-3-flash-preview",
            messages=messages,
            max_tokens=5
        )

        

        try:
            content = response.choices[0].message.content.strip().upper().translate(str.maketrans('', '', string.punctuation))
            print(content)
        except:
            print("Error processing response:", response.choices[0].message.content)
            break # skip the rest of this notebook due to invalid response on current cell
    
        if content == "YES":
            is_buggy = 1
        elif content == "NO":
            is_buggy = 0
        else:
            skipped += 1
            continue

        if is_buggy != label:
            correctnessOfPredictions.append(False)
        else:
            correctnessOfPredictions.append(True)

    except Exception as e:
        # print(e)
        skipped += 1
        continue

    results.append((is_buggy, label))


    preds_so_far = [pred for pred, _ in results]
    labels_so_far = [true for _, true in results]
    f1 = f1_score(labels_so_far, preds_so_far, zero_division=0)
    acc = accuracy_score(labels_so_far, preds_so_far)
    tq.set_postfix({'F1': f"{f1:.3f}", 'Acc': f"{acc:.3f}", 'Recall': f"{recall_score(labels_so_far, preds_so_far, zero_division=0):.3f}"})
    tq.refresh()  




correct = sum([pred == true for pred, true in results])
total = len(results)
if total != 0:
    accuracy = correct / total


predictions = [pred for pred, _ in results]
labels = [true for _, true in results]

# compute metrics
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)
f1 = f1_score(labels, predictions)

print(f"Final Accuracy: {accuracy:.4f}")
print(f"Final Precision: {precision:.4f}")  
print(f"Final Recall: {recall:.4f}")
print(f"Final F1 Score: {f1:.4f}")

tp = 0
fp = 0
tn = 0
fn = 0

for i, (pred, true) in enumerate(zip(predictions, labels)):
    if pred == 1 and true == 1:
        tp += 1
    elif pred == 1 and true == 0:
        fp += 1
    elif pred == 0 and true == 0:
        tn += 1
    elif pred == 0 and true == 1:
        fn += 1


print(f"\nTrue Positives:  {tp}")
print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"Skipped: {skipped}")

