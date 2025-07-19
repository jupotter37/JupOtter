import torch
from torch.amp import autocast
from tqdm.auto import tqdm

class VectorEval:
    def __init__(self):
        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        self.true_negatives = 0
        self.total_books = 0
        self.total_1 = 0
        self.total_0 = 0

        self.all_f1_scores = []

        # These values will hold the calculated averages on a per-book basis
        self.average_percent_bugs_found_per_book = 0
        self.average_precision_score = 0
        self.average_recall_score = 0
        self.average_f1_score = 0
        self.average_accuracy_score = 0

        self.total_bugs = 0
        self.total_cells = 0

    '''
    Used as a helper function in cell-level evaluation to evaluate prediction vectors agains their labels.
    '''
    def eval_vector(self, result_vector, label_vector):
        # Ensure both are iterables
        if isinstance(result_vector, int):
            result_vector = [result_vector]
        if isinstance(label_vector, int):
            label_vector = [label_vector]
        
        local_true_positives = 0
        local_false_positives = 0
        local_true_negatives = 0
        local_false_negatives = 0
        local_buggy_cells = 0
        local_non_buggy_cells = 0

        for prediction, label in zip(result_vector, label_vector):
            if prediction == 1 and label == 1:  # True Positive
                local_true_positives += 1
                local_buggy_cells += 1
            elif prediction == 1 and label == 0:  # False Positive
                local_false_positives += 1
                local_non_buggy_cells += 1
            elif prediction == 0 and label == 0:  # True Negative
                local_true_negatives += 1
                local_non_buggy_cells += 1
            elif prediction == 0 and label == 1:  # False Negative
                local_false_negatives += 1
                local_buggy_cells += 1
            if prediction == 1:
                self.total_1 += 1
            else:
                self.total_0 += 1

        local_total_cells = local_buggy_cells + local_non_buggy_cells

        local_precision = self.calculate_precision(local_true_positives, local_false_positives)
        local_recall = self.calculate_recall(local_true_positives, local_false_negatives)
        local_f1_score = self.calculate_f1_score(local_precision, local_recall)
        self.all_f1_scores.append(local_f1_score)
        local_accuracy = self.calculate_accuracy(local_true_positives, local_true_negatives,
                                                 local_false_positives, local_false_negatives)
        # When no buggy cells, define percent bugs found as 1 (i.e., no bugs missed)
        local_average_percent_bugs_found_per_book = (local_true_positives / local_buggy_cells) if (local_buggy_cells > 0) else 1

        # Update the weighted averages per book
        self.average_precision_score = ((self.average_precision_score * self.total_books) + local_precision) / (self.total_books + 1)
        self.average_recall_score = ((self.average_recall_score * self.total_books) + local_recall) / (self.total_books + 1)
        self.average_f1_score = self.calculate_f1_score(self.average_precision_score, self.average_recall_score)
        self.average_accuracy_score = ((self.average_accuracy_score * self.total_books) + local_accuracy) / (self.total_books + 1)
        self.average_percent_bugs_found_per_book = ((self.average_percent_bugs_found_per_book * self.total_books) + local_average_percent_bugs_found_per_book) / (self.total_books + 1)
        self.total_books += 1

        # Also update the global counters
        self.true_positives += local_true_positives
        self.false_positives += local_false_positives
        self.true_negatives += local_true_negatives
        self.false_negatives += local_false_negatives
        self.total_bugs += local_buggy_cells
        self.total_cells += local_total_cells

    '''
    Used a helper function in file-level evaluation to evaulate model predictions agains the label.

    This is for file-level predictions, so if a cell in the label is labeled as buggy the label becomes
    1, and if a cell is predicted to be buggy in the model outputs the prediction becomes 1.
    '''
    
    def eval_vector_file(self, result_vector, label_vector):
        label = 0

        final_prediction = 0
        for prediction in result_vector:
            if prediction == 1:
                final_prediction = 1
                break
        
        for item in label_vector:
            if item == 1:
                label = 1
                break
        if final_prediction == 1 and label == 1:  # True Positive
            self.true_positives += 1
        elif final_prediction == 1 and label == 0:  # False Positive
            self.false_positives += 1
        elif final_prediction == 0 and label == 0:  # True Negative
            self.true_negatives += 1
        elif final_prediction == 0 and label == 1:  # False Negative
            self.false_negatives += 1
        if final_prediction == 1:
            self.total_1 += 1
        else:
            self.total_0 += 1

    '''
    This function evaluates a single book (one sample) given its predictions and labels.
    It will print the logits predictions for each cell in the book, and use eval_vector()
    to evaluate.
    '''
    def eval_single_book(self, test_loader, model, start_token_ids, end_token_ids, device, chunk_size=4, eval_type=1):
        model.eval()
        print(f"Number of samples in test_loader dataset: {len(test_loader.dataset)}")

        # Get first batch from DataLoader iterator
        batch = next(iter(test_loader))
 
        input_ids_batch = batch['input_ids'][0][:chunk_size].to(device)
        attention_mask_batch = batch['attention_mask'][0][:chunk_size].to(device)
        batch_labels = [lbl.to(device) for lbl in batch['labels'][0][:chunk_size]]

        label = torch.cat(batch_labels).int().tolist()

        with torch.no_grad():
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch,
                    start_token_ids=start_token_ids,
                    end_token_ids=end_token_ids,
                    labels=None
                )

        logits_pred = (torch.cat(outputs["logits"]).squeeze(1) >= 0).long().cpu().tolist()
        print("Label:  ", label)
        print("Logits: ", logits_pred)
    
        self.eval_vector(logits_pred, label)
     

    '''
    The following function is called durring training and evaluation to evaluate models. Eval type 1 is for cell 
    level bug detection, eval type 2 is for file level bug detection.

    Evaluates the model over the test set provided by test_loader in batches.

    This will evaluate the model one book at a time, and for each book, it will evaluate the model on all its chunks 
    predictions concatenated together to form a single tensor for the notebook.
    '''
    def eval_vector_batched(self, test_loader, model, start_token_ids, end_token_ids, device, chunk_size=4, eval_type=1):
        model.eval()

        loader = tqdm(test_loader,
                  desc="Evaluating batches",
                  total=len(test_loader),
                  unit="batch")

        with torch.no_grad():
            for batch in loader:
                input_ids_batch = batch['input_ids']
                attention_mask_batch = batch['attention_mask']
                labels_batch = batch['labels']

                for batch_ids, batch_masks, batch_labels in zip(input_ids_batch, attention_mask_batch, labels_batch):# looping for each item in the batch to get results of its chunks
                    
                    batch_ids = batch_ids[:chunk_size].to(device)  # may be multiple chunks because each notebook can have multiple chunks, and this is a notebook
                    batch_masks = batch_masks[:chunk_size].to(device)  # Move each mask tensor to the device
                    batch_labels = [lbl.to(device) for lbl in batch_labels[:chunk_size]]  # Move each label tensor to the device
        
                    
                    with autocast(device_type='cuda', dtype=torch.float16): 
                        outputs = model(input_ids=batch_ids, attention_mask=batch_masks, start_token_ids=start_token_ids, end_token_ids=end_token_ids, labels=batch_labels)
                    
                    # concatenate the predictions together to form a single tensor for the notebook
                    logits_pred = (torch.cat(outputs["logits"]).squeeze(1) >= 0).long().cpu().tolist()

                    # concatenate the chunk level labels
                    label = torch.cat(batch_labels).float().tolist()
     
                    if eval_type == 1 and label and logits_pred is not None: # cell level eval
                        self.eval_vector(logits_pred, label)
  
                    if eval_type == 2: # file level eval
                        self.eval_vector_file(logits_pred, label)



                      
    '''
    Helper metric functions
    '''
    def calculate_precision(self, true_positives, false_positives):
        return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1

    def calculate_recall(self, true_positives, false_negatives):
        return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 1

    def calculate_f1_score(self, precision, recall):
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def calculate_accuracy(self, true_positives, true_negatives, false_positives, false_negatives):
        total = true_positives + true_negatives + false_positives + false_negatives
        return (true_positives + true_negatives) / total if total > 0 else 1

    def get_total_precision(self):
        return self.calculate_precision(self.true_positives, self.false_positives)
       
    def get_total_recall(self):
        return self.calculate_recall(self.true_positives, self.false_negatives)
    
    def get_total_f1(self):
        p = self.get_total_precision()
        r = self.get_total_recall()
        return self.calculate_f1_score(p, r)
    
    def get_total_accuracy(self):
        return self.calculate_accuracy(self.true_positives, self.true_negatives, self.false_positives, self.false_negatives)

    def get_total_buggy_cell_ratio(self):
        return self.total_bugs / self.total_cells if self.total_cells > 0 else 1

    '''
    The following functions are used to print the results of the evaluation
    '''
    def print_results(self):
        print(f"Total 1: {self.total_1}, Total 0: {self.total_0}")
        print("\n===== Model Evaluation Metrics Totals =====")
        print(f"Total books: {self.total_books}, Total cells: {self.total_cells}, Total buggy cells: {self.total_bugs}")
        print(f"True positives: {self.true_positives}, True negatives: {self.true_negatives}, Total correct: {(self.true_negatives + self.true_positives)}")
        print(f"False positives: {self.false_positives}, False negatives: {self.false_negatives}, Total incorrect: {(self.false_negatives + self.false_positives)}")
        print()
        print("===== Cell-aggregated =====")
        print(f"Precision: {self.get_total_precision():.4f}")
        print(f"Recall: {self.get_total_recall():.4f}")
        print(f"F1 Score: {self.get_total_f1():.4f}")
        print(f"Accuracy: {self.get_total_accuracy():.4f}")
        print(f"Buggy Cell Ratio: {self.get_total_buggy_cell_ratio():.4f}")
        print("===================================")
        print()
        print("\n===== File-aggregated =====")
        print(f"Precision Score: {self.average_precision_score:.4f}")
        print(f"Recall Score: {self.average_recall_score:.4f}")
        print(f"F1 Score: {self.average_f1_score:.4f}")
        print(f"Accuracy Score: {self.average_accuracy_score:.4f}")
        print("===================================")

    def print_results_file_level(self):
        print(f"Total 1: {self.total_1}, Total 0: {self.total_0}")
        print("\n===== Model File Level Evaluation Metrics Totals =====")
        print(f"True positives: {self.true_positives}, True negatives: {self.true_negatives}, Total correct: {(self.true_negatives + self.true_positives)}")
        print(f"False positives: {self.false_positives}, False negatives: {self.false_negatives}, Total incorrect: {(self.false_negatives + self.false_positives)}")
        print(f"Precision: {self.get_total_precision():.4f}")
        print(f"Recall: {self.get_total_recall():.4f}")
        print(f"F1 Score: {self.get_total_f1():.4f}")
        print(f"Accuracy: {self.get_total_accuracy():.4f}")
        print("===================================")

    '''
    To reset the evaluation metrics between evaluations.
    '''
    def reset(self):
        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        self.true_negatives = 0
        self.total_books = 0

        # These values will hold the calculated averages on a per-book basis
        self.average_percent_bugs_found_per_book = 0
        self.average_precision_score = 0
        self.average_recall_score = 0
        self.average_f1_score = 0
        self.average_accuracy_score = 0

        self.total_bugs = 0
        self.total_cells = 0

        self.total_1 = 0
        self.total_0 = 0
        
        self.all_f1_scores = []