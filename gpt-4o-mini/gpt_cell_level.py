import openai
import torch
import time
import os
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re
from transformers import RobertaTokenizer
import string
import buggy_cell_vector_evalualtion_clean as eval

client = OpenAI(api_key="")
load_path = "L:\\base_2500.pt"# path to saved tonkenized data
# load_path = r"G:\JupOtter\dataset\tokenized_content\parrot.pt" # path to saved tonkenized data
# load_path = r"G:\JupOtter\dataset\tokenized_content\je.pt" # path to saved tonkenized data
tokenized_data = torch.load(load_path)

#the following us used to load OtterDataset (split data)
train_ids = tokenized_data['train_ids']
test_ids = tokenized_data['test_ids']
train_masks = tokenized_data['train_masks']
test_masks = tokenized_data['test_masks']
train_labels = tokenized_data['train_labels']
test_labels = tokenized_data['test_labels']


# Uncomment the lines below to load unsplit data
# test_ids = tokenized_data['test_ids']
# test_masks = tokenized_data['test_masks']
# test_labels = tokenized_data['test_labels']


# #shuffling done for parrot and jupyter errors. JupOtter is already shuffled.
# import random

# random.seed(42)

# combined = list(zip(test_ids, test_masks, test_labels))
# random.shuffle(combined)

# test_ids, test_masks, test_labels = zip(*combined)

# test_ids = list(test_ids)
# test_masks = list(test_masks)
# test_labels = list(test_labels)
# #end shuffling

print("Tokenized data loaded successfully.")


tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

# setting up the special tokens use for finding cell boundaries in tokenized content
start_special_tokens = [f"<CELL_{i}>" for i in range(1, 1024)]
end_special_tokens = [f"<END_CELL_{i}>" for i in range(1, 1024)]
all_special_tokens = start_special_tokens + end_special_tokens


for token in all_special_tokens:
    if token not in tokenizer.get_vocab():
        tokenizer.add_tokens([token])


model_tester = eval.VectorEval()
model_tester2 = eval.VectorEval()


flat_codes, flat_labels = [], []

for chunks_ids, chunks_masks, chunk_label_lists in tqdm(
    zip(test_ids[:100], test_masks[:100], test_labels[:100]),
    total=len(test_ids),
    desc="Decoding & cleaning notebooks",
    dynamic_ncols=True,
):
    file_ids = chunks_ids[:4] # we use 4 because that is the same number of chunks used when evaluating JupOtter
    chunks_label = chunk_label_lists[:4]
  
    flat_list = file_ids.reshape(-1).tolist() 

    decoded_with_cells = tokenizer.decode(flat_list, skip_special_tokens=False)



    decoded_clean = decoded_with_cells
    for token in tokenizer.all_special_tokens:
        pattern = re.escape(token)
        decoded_clean = re.sub(pattern, "", decoded_clean)
    
    cells = re.split(r"<CELL_\d+>", decoded_clean)
    cells = [re.sub(r"<END_CELL_\d+>", "", c).strip() for c in cells if len(c.strip()) > 0]


    flat_codes.append(cells)
    flat_labels.append([int(item.item()) for sublist in chunks_label for item in sublist])

    # check lengths
    if len(cells) != len(flat_labels[-1]):
        print("Length mismatch!")
        print("Cells:", len(cells))
        print("Labels:", len(flat_labels[-1]))



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
for i, (cells, cell_labels) in tq:
    prediction = []
    messages = [
        {
            "role": "system",
            "content": (
                "You are analyzing a Jupyter notebook cell-by-cell. "
                "You must remember previous cells. "
                "For each cell, answer ONLY with YES or NO indicating whether THIS CELL contains a bug."
            )
        }
    ]

    for cell in cells:

        messages.append({
            "role": "user",
            "content": f"Here is the next cell:\n\n{cell}\n\nDoes THIS cell contain a bug?"
        })

        try:
            response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=5
        )
            
            try:
                content = response.choices[0].message.content.strip().upper().translate(str.maketrans('', '', string.punctuation))
                
            except:
                print("Error processing response:", response.choices[0].message.content)
                break 

            if content == "YES":
                is_buggy = 1
            elif content == "NO":
                is_buggy = 0
            else:
                # skipped += 1
                print("Invalid response from model:", content)
            
                break 

            prediction.append(is_buggy)

            messages.append({
                "role": "assistant",
                "content": content
            })
            
        except Exception as e:
            print("Error during model inference:", str(e))
            
            time.sleep(1)
            break
        time.sleep(1)

    results.append((prediction, cell_labels))
    if len(prediction) != len(cell_labels): # get the label from the list, it is a tensor wrapped in a list
        print("Skipping due to length mismatch")
        print("len prediction:", len(prediction))
        print("len cell_labels:", len(cell_labels))
        skipped += 1
        model_tester2.eval_vector(prediction, cell_labels[:len(prediction)])
        continue
    
    else:
        model_tester.eval_vector(prediction, cell_labels)
        model_tester2.eval_vector(prediction, cell_labels)
    
    if (i + 1) % 10 == 0:
        torch.save(results, "gpt_cell_level_results_Otter_100.pt")


   
    f1 = model_tester.average_f1_score

    tq.set_postfix({'F1': f"{f1:.3f}"})
    tq.refresh()  


torch.save(results, "gpt_cell_level_results_Otter_100.pt")
model_tester.print_results()
print()
print()

model_tester2.print_results()
print(f"Skipped: {skipped}")
