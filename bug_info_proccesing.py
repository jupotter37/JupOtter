import re
import ast
import nbformat
from IPython.core.inputtransformer2 import TransformerManager


'''
The following function is used to check if a notebook is buggy, by checking for syntax errors and runtime errors.
Syntax errors are checked by parsing the code in each cell to ensure errors are caught in cells with no cell outputs 
(parsing code results in 6% more errors caught that would otherwise be classified as not buggy),
and runtime errors are checked by looking at the outputs of each cell.

Parsing is done using parse_idv_cells(). Notebooks are parsed at the cell level because syntax is independant of 
surrounding cells.

Runtime errors are checked using get_runtime_errors().
'''
def check_if_notebook_is_buggy(notebook_file_path):
    # open notebook
    with open(notebook_file_path, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)
        
    cell_count = 1
    error_locations, error_types, cleaned_error_messages, buggy_cells = [], [], [], []
    isBug = False

    # loop through the code cells
    for cell in notebook_content.cells: 
        if cell.cell_type == 'code':
            code = cell['source']
                              
            # to find syntax errors 
            isSyntaxError, syntax_error_message, syntax_error_line, syntax_error_location  = parse_idv_cells(code, cell_count)

            if isSyntaxError == True:
                isBug = True
                error_types.append('SyntaxError')
                cleaned_error_messages.append(syntax_error_message)
                error_locations.append(syntax_error_location)
                buggy_cells.append(1)
            else: # looking for runtime error saved in book data
                isRuntimeError, runtime_error_type, runtime_error_message, runtime_error_line, runtime_error_location = get_runtime_errors(cell, cell_count)
                if isRuntimeError and runtime_error_type not in {'ImportError', 'FileNotFoundError', 'KeyboardInterrupt', 'SystemExit', 'ConnectionError'}:
                    isBug = True
                    error_types.append(runtime_error_type)
                    cleaned_error_messages.append(runtime_error_message)
                    error_locations.append(runtime_error_location)
                    buggy_cells.append(1)
                else:
                    buggy_cells.append(0) # to indicate not buggy
            cell_count += 1 # indent one more level if want to count all cells not just code
        
    # return info about if book is buggy
    if isBug:
        return 1, error_locations, error_types, cleaned_error_messages, buggy_cells
    else:
        return 0, "No errors", "No errors", "No errors", buggy_cells
    

'''
This function checks for runtime errors in a cell. If an error is found (ename) then the cell is considered buggy
and the error type, message, line number and location are returned.
'''
def get_runtime_errors(cell, cell_count):
    isBug = False

    # defaults
    error_type = None  
    error_message = None
    error_line = None  
    error_location_build = None

    # check for outputs that are errors, this is for finding all errors other than syntax
    for output in cell.get('outputs', []):
        if 'ename' in output:  # If there is an error
            # bug is found
            isBug = True
            # get the error type
            error_type = output.get('ename', 'Unknown error type').strip()

            error_location_build = error_type + " in cell " + str(cell_count)

            # get error message and just the error message
            error_message = output.get('evalue', 'Unknown error')                     

            # search error message for where error occured
            line_match = re.search(r'line (\d+)', error_message)
            if line_match:
                error_line = line_match.group(1)  # extract the line number
                error_location_build += ", on line " + error_line
            # if didnt find line in error message search traaceback
            else:
                found = False
                traceback_lines = output.get('traceback', [])
                # loop through the traceback looking for error line
                for line in traceback_lines:
                    # Look for a line that mentions line number
                    match = re.search(r'line (\d+)', line)
                    if match:
                        error_line = match.group(1)  # extract the line number
                        error_location_build += ", on line " + error_line
                        found = True
                        break  # stop after finding the line number      
                # if did not find put that in the data
                if found == False:         
                    error_location_build += ", line not found"     

    return isBug, error_type, error_message, error_line, error_location_build

'''
This function parses a single cell of code to check for syntax errors using the ast module.

It returns a tuple indicating whether a syntax error was found, the error message, the line number
'''
def parse_idv_cells(code, cell_count):

    transformer = TransformerManager()

    try:
        # transforms ipynb code to py code to avoid false positives when parsing
        transformed_code = transformer.transform_cell(code)
        # try parsing the code, if this fails caught syntax error in except
        ast.parse(transformed_code)
                        
    except SyntaxError as e:
        error_line = e.lineno
        error_message = f"Syntax error: {e.msg}"
        error_location_build = f"SyntaxError in cell {cell_count}, on line " + str(error_line)
        return True, error_message, error_line, error_location_build
            
    return False, '', '', '' # default, no error