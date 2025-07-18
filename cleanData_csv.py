import csv
import os
import ast
import code_proccesing
import bug_info_proccesing



class Data_Clean():
    '''
    The following function is used to process a folder of notebooks creating a csv file with annotations about the notebooks.
    '''
    def process_notebooks(self, input_notebooks, writeToFileName, Format=2):
        # Open a CSV file to store the results
        with open(writeToFileName, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['notebook_name', 'notebook_structure', 'is_buggy', 'error_locations', 'error_type', 'error_message', 'buggy_cells']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            bookCount = 0
            # Process each notebook
            for notebook_file_path in input_notebooks:
                # Extract the code cells from the notebook
                try:
                    # Determine if the notebook is buggy, looks at file
                    isNoteBookBuggy, error_locations, error_type, error_message, buggy_cells =  bug_info_proccesing.check_if_notebook_is_buggy(notebook_file_path)

                    # determine what type of code format is desired from code_proccesing functions
                    if Format == 1:
                        new_notebook = code_proccesing.extract_cells_with_cell_nums(notebook_file_path)
                    elif Format == 2:
                        new_notebook = code_proccesing.extract_code_one_file(notebook_file_path) # what was used for OtterDataset
                    elif Format == 3:
                        new_notebook = code_proccesing.extract_code_and_markdown_one_file(notebook_file_path)
                    elif Format == 4:
                        new_notebook = code_proccesing.extract_cells_bracketed(notebook_file_path)

                    notebook_file_path_array = notebook_file_path.split('/')
                    notebook_name = notebook_file_path_array[-1]       
                    
                    writer.writerow({
                        'notebook_name': notebook_name,
                        'notebook_structure': new_notebook,
                        'is_buggy': isNoteBookBuggy,
                        'error_locations': error_locations,
                        'error_type' : error_type,
                        'error_message': error_message,
                        'buggy_cells': buggy_cells
                    })
                    
                    bookCount += 1
     
                except Exception as e:
                    print('Error opening: ' + notebook_file_path + "\n")
                    print(e)

    '''
    Used to process a single notebook, used for testing and debugging.
    '''
    def process_single_book(self, notebook_file_path, write_to_file_name, format_type=1):
        try:
            # Open a CSV file to store the results
            with open(write_to_file_name, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['notebook_name', 'notebook_structure', 'is_buggy', 'error_locations', 'error_type', 'error_message', 'buggy_cells']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                # Check if the notebook is buggy
                
                is_notebook_buggy, error_locations, error_type, error_message, buggy_cells = bug_info_proccesing.check_if_notebook_is_buggy(notebook_file_path)
                print(1)
                # Determine the format
                if format_type == 1:
                    new_notebook = code_proccesing.extract_cells_with_cell_nums(notebook_file_path)
                elif format_type == 2:
                    new_notebook = code_proccesing.extract_code_one_file(notebook_file_path)
                elif format_type == 3:
                    new_notebook = code_proccesing.extract_cells_raw(notebook_file_path)
                elif format_type == 4:
                    new_notebook = code_proccesing.extract_cells_bracketed(notebook_file_path)
                else:
                    raise ValueError(f"Invalid format type: {format_type}")

                # Extract notebook name
                notebook_name = notebook_file_path.split('/')[-1]

                # Write the data to the CSV file
                writer.writerow({
                    'notebook_name': notebook_name,
                    'notebook_structure': new_notebook,
                    'is_buggy': is_notebook_buggy,
                    'error_locations': error_locations,
                    'error_type' : error_type,
                    'error_message': error_message,
                    'buggy_cells': buggy_cells
                })
                print(f"Processed notebook: {notebook_name}")

        except Exception as e:
            print(f"Error processing notebook: {notebook_file_path}")
            print(f"Exception: {e}")

    
    # Creates a list of all files in the directory with their full paths
    def get_file_names(self, directory):
        return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    def get_file_name(self, file_path):
        if os.path.isfile(file_path):
            return file_path
        else:
            raise ValueError(f"The file {file_path} does not exist or is not a valid file.")

    '''
    Used to create training data for JupOtter, it uses process_notebooks to process all notebooks in a directory.
    '''
    def create_notebook_train_data(self, directory, writeToFileName, Format):
        list_of_notebooks = self.get_file_names(directory)
        self.process_notebooks(list_of_notebooks, writeToFileName, Format)

    '''
    Used to create csv with single book data, this ment to be ran on a single ipynb file, put path of file
    csv name, then format.
    '''
    def create_single_notebook_data(self, directory, writeToFileName, Format):
        name = self.get_file_name(directory)
        self.process_single_book(name, writeToFileName, Format)

class Data_manage:
    '''
    Used to count errors that have been idendified in the csv column containing the error types.
    '''
    def countErrorTypes(self, FileName):
        # this limit is used to avoid crashing with large notebooks
        csv.field_size_limit(1000000)
        errorTypes = {}

        with open(FileName, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile) 
            # going through each row of data
            for row in reader:
                error_messages = row[-3].strip() # get this rows error types
                
                # If the error messages are in a string format like a list, convert it to a list
                try:
                    error_messages_list = eval(error_messages)
                    if isinstance(error_messages_list, list):  # Check if it's a valid list

                        for error_message in error_messages_list:
                            if error_message in errorTypes:
                                errorTypes[error_message] += 1  # increment count if error exists
                            else:
                                errorTypes[error_message] = 1  # initialize key with count 1
                    else:# create error to go in the except if the list was not valid
                        raise ValueError("Parsed error_messages is not a list") 
                except Exception as e:
                    # this will mainly add 'No errors' to the list of errors. 
                    error_messages = error_messages.strip()
                    if error_messages in errorTypes:
                        errorTypes[error_messages] += 1  # Increment count if error exists
                    else:
                        errorTypes[error_messages] = 1  # Initialize key with count 1
        
        # Print the dictionary formatted for readability so can see errors
        sorted_error_types = sorted(errorTypes.items(), key=lambda item: item[1])
        for error_type, count in sorted_error_types:
            print(f"{error_type}: {count}\n")

    '''
    To determine how many buggy and non buggy files there are
    '''
    def bugRatio(self, FileName):
        csv.field_size_limit(10000000) 
        buggy = 0
        notBuggy = 0
        with open(FileName, 'r', newline='', encoding='utf-8') as csvfile:
            fileReader = csv.reader(csvfile)
            next(fileReader) 
            for row in fileReader:
                if row[2] == "1":
                    buggy += 1
                else:
                    notBuggy += 1
        print(f"Not buggy: {notBuggy} Buggy: {buggy}")

    def bugRatioCellLevel(self, FileName):
        csv.field_size_limit(10000000) 
        buggy = 0
        notBuggy = 0
        with open(FileName, 'r', newline='', encoding='utf-8') as csvfile:
            fileReader = csv.reader(csvfile)
            next(fileReader) # skipp header
            for row in fileReader:
                array = ast.literal_eval(row[-1])
                for i in array:
                    if i == 1:
                        buggy += 1
                    else:
                        notBuggy += 1
        print(f"Not buggy: {notBuggy} Buggy: {buggy}")

    '''
    Returns true if all entries within a given csv are unique based on the name, otherwise false. This is
    used to prevent data leakage, name is a combination the notebook name, author name, and repository name.
    '''
    def ensureNoDuplicates(self, FileName):
        csv.field_size_limit(10000000)
        with open(FileName, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)  # Use CSV reader to parse rows
            for row in reader:
                for potentialDuplicateRow in reader:
                    if row[0] == potentialDuplicateRow[0]:
                        return False
        return True
    
    '''
    Returns true if all entries between two csv files are unique, returns true if all entries are unique,
    otherwise returns false.
    '''
    def ensure_no_duplicates_between_files(self, file_a, file_b):
        csv.field_size_limit(10000000)

        seen = set()

        with open(file_a, newline='', encoding='utf-8') as fa:
            reader = csv.reader(fa)
            next(reader)  # Skip header
            for row in reader:
                seen.add(row[0])

        # Check each key in the second file against the first
        with open(file_b, newline='', encoding='utf-8') as fb:
            reader = csv.reader(fb)
            next(reader)  # Skip header
            for row in reader:
                if row[0] in seen:
                    # Found a duplicate
                    return False
        return True
    
    '''
    Used to remove duplicates between 2 csv files. The duplicates will be removed from file_b.
    '''
    def remove_duplicates_from_file_b_in_place(self, file_a, file_b):
        csv.field_size_limit(10000000)

        # read all unique keys from file_a into a set
        seen_in_file_a = set()
        with open(file_a, newline='', encoding='utf-8') as fa:
            reader = csv.reader(fa)
            next(reader)  # Skip header
            for row in reader:
                if row:
                    seen_in_file_a.add(row[0])

        temp_file_b = file_b + '.tmp'

        # read file_b and write only unique rows directly to temp file
        with open(file_b, newline='', encoding='utf-8') as fb, \
            open(temp_file_b, 'w', newline='', encoding='utf-8') as temp_fb:
            reader = csv.reader(fb)
            writer = csv.writer(temp_fb)
            # Copy header unchanged
            header = next(reader)
            writer.writerow(header)
            for row in reader:
                if row and row[0] not in seen_in_file_a:
                    writer.writerow(row)

        # replace original file with temp file
        os.replace(temp_file_b, file_b)


    def count_num_entries(self, dir):
        csv.field_size_limit(1000000) 
        total = 0
        for filename in os.listdir(dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(dir, filename)

                a = self.ensureNoDuplicates(file_path)
                print(f"File: {filename}, Dupluicates: {a}")
                with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile)  # Use CSV reader to parse rows
                    count = -1 # to skipp header
                    for row in reader:
                        count += 1
                    print(count)
                    total += count
        print(f"Total: {total}")

    def count_csv_rows(self, file_path):
        csv.field_size_limit(1000000) 
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                next(reader, None)  # Read the header row
                row_count = 0
                for row in reader:
                    row_count += 1
            return row_count
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            return -1
        except Exception as e:
            print(f"An error occurred: {e}")
            return -1

class Move_books:
    '''
    The following function is used to move notebooks from one csv to another, it will only move the buggy notebooks,
    and will not produce duplicates.
    '''
    def moveBuggyCode(self, MoveToFile, MoveFromFile):
        buggyRows = []

        csv.field_size_limit(1000000) 
        
        with open(MoveFromFile, 'r', encoding='utf-8') as csvfile:
            fromFileReader = csv.reader(csvfile)
            for row in fromFileReader:
                if row[2] == "1":
                    buggyRows.append(row)

        totalMoved = 0
        # Check if the destination file exists
        if not os.path.exists(MoveToFile):
            # Create a new file if it does not exist
            with open(MoveToFile, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['notebook_name', 'notebook_structure', 'is_buggy', 'error_locations', 'error_type', 'error_message', 'buggy_cells']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for row in buggyRows:
                    writer.writerow({
                                'notebook_name': row[0],
                                'notebook_structure': row[1],
                                'is_buggy': row[2],
                                'error_locations': row[3],
                                'error_type': row[4],
                                'error_message': row[5],
                                'buggy_cells': row[6]
                    })  
                    print(f"Created new file and added buggy rows to {MoveToFile}")
        else:
            # If file exists, append to it
            with open(MoveToFile, 'r', encoding='utf-8') as csvfile:
                toFileReader = csv.reader(csvfile)
                existingRows = [row for row in toFileReader]
            
            # Open the file in append mode
            with open(MoveToFile, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Append only rows that don't already exist in the destination file
                for buggyRow in buggyRows:
                    if self.checkIfContains(existingRows, buggyRow[0]):
                        totalMoved += 1
                        writer.writerow(buggyRow)
            print(f"Appended new buggy rows to {MoveToFile}, moved {totalMoved} rows")

    '''
    The following function is used to move notebooks from one csv to another, not only the buggy notebooks.
    '''
    def moveCode(self, MoveToFile, MoveFromFile):
        buggyRows = []

        csv.field_size_limit(1000000) 
        
        with open(MoveFromFile, 'r', encoding='utf-8') as csvfile:
            fromFileReader = csv.reader(csvfile)
            for row in fromFileReader:
                buggyRows.append(row)

        totalMoved = 0
        # Check if the destination file exists
        if not os.path.exists(MoveToFile):
            # Create a new file if it does not exist
            with open(MoveToFile, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['notebook_name', 'notebook_structure', 'is_buggy', 'error_locations', 'error_type', 'error_message', 'buggy_cells']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for row in buggyRows:
                    writer.writerow({
                                'notebook_name': row[0],
                                'notebook_structure': row[1],
                                'is_buggy': row[2],
                                'error_locations': row[3],
                                'error_type': row[4],
                                'error_message': row[5],
                                'buggy_cells': row[6]
                    })  
                    print(f"Created new file and added buggy rows to {MoveToFile}")
        else:
            # If file exists, append to it
            with open(MoveToFile, 'r', encoding='utf-8') as csvfile:
                toFileReader = csv.reader(csvfile)
                existingRows = [row for row in toFileReader]
            
            # Open the file in append mode
            with open(MoveToFile, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Append only rows that don't already exist in the destination file
                for buggyRow in buggyRows:
                    if self.checkIfContains(existingRows, buggyRow[0]):
                        totalMoved += 1
                        writer.writerow(buggyRow)
            print(f"Appended new buggy rows to {MoveToFile}, moved {totalMoved} rows")
            # number of entries in move from file
            print(f"Number of entries in {MoveFromFile}: {len(buggyRows)}")
    
    '''
    checks if a notebook is already in a csv, ensures no duplicates
    '''
    def checkIfContains(self, toFileReader, buggyCodeName):
        for row in toFileReader:
            if row[0] == buggyCodeName:
                return False
        return True
