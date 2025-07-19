import re
import nbformat

'''
The following function is used to exrtract only code cells from a notebook.
This will return the cells as a string in the format [{cell_number n, code source code}].

Not used in OtterDataset
'''
def extract_cells_bracketed(notebook_file_path):
        # Load the notebook using nbformat
        with open(notebook_file_path, 'r', encoding='utf-8') as f:
            notebook_content = nbformat.read(f, as_version=4)

        # Initialize list to store code cells with numbers
        code_cells = []
    
        # Loop through each cell in the notebook and extract the code
        for idx, cell in enumerate(notebook_content.cells, start=0):
            if cell.cell_type == 'code':
                source = cell['source']

                # Remove single-line comments
                source = re.sub(r'#.*', '', source)

                # Remove multi-line comments (triple quotes)
                source = re.sub(r"'''(.*?)'''", '', source, flags=re.DOTALL)

                # Clean up any trailing or leading whitespaces, including newlines
                source = source.strip()


                # Store each code cell as a dictionary with cell number and source code
                code_cells.append({
                    'cell_number': idx,
                    'code': source
                })
        
        return code_cells

'''
This will extract the code from a notebook, adding markers and removing comments in the cells.

Not used in OtterDataset
'''
# used to get the content out of a single notebook organized by cells
def extract_cells_with_cell_nums(notebook_file_path):
        # Load the notebook using nbformat
        with open(notebook_file_path, 'r', encoding='utf-8') as f:
            notebook_content = nbformat.read(f, as_version=4)

        # Loop through each cell in the book and take out the source code
        code_sources = []
        cell_num = 1
        for cell in notebook_content.cells:
             if cell.cell_type == 'code':
                cell_num += 1
                source = cell['source']
                
                # Remove single-line comments
                source = re.sub(r'#.*', '', source) 

                # Remove multi-line comments (triple quotes)
                source = re.sub(r"'''(.*?)'''", '', source, flags=re.DOTALL)  
                # Remove any leading/trailing whitespace, including newlines
                source = source.strip()
                # Append cleaned code to the list
                code_sources.append("Code cell: " + str(cell_num) + '\n' + source)
                 
        # Return the list of code cells
        return code_sources

'''
This will extract the code from a notebook and return it as a string seperated
by <CELL_1> <END_CELL_1>, <CELL_2>, <END_CELL_2>, etc.

This was the format used in OtterDataset.
'''
def extract_code_one_file(notebook_file_path):
        # Load the notebook using nbformat
        with open(notebook_file_path, 'r', encoding='utf-8') as f:
            notebook_content = nbformat.read(f, as_version=4)
        cell_count = 1
        # Loop through each cell in the book and take out the source code
        code_sources = ""
        for cell in notebook_content.cells:
            if cell.cell_type == 'code':
                source = cell['source']
                
                cell_pre = f"<CELL_{cell_count}>\n"
                cell_finish = f"\n<END_CELL_{cell_count}>\n\n"
                cell_count += 1
                
                # Append cell identifier and cleaned source code
                code_sources += cell_pre + source + cell_finish

        return code_sources
'''
This will extract the code and markdown from a notebook and return it as a string seperated
by <CODE_CELL_1> <END_CODE_CELL_1>, <CODE_CELL_2>, <END_CODE_CELL_2>, etc for code cells, and 
<MARKDOWN_CELL_1> <END_MARKDOWN_CELL_1>, <MARKDOWN_CELL_2>, <END_MARKDOWN_CELL_2>, etc for markdown cells.

Not used in OtterDataset
'''
def extract_code_and_markdown_one_file(notebook_file_path):
        # Load the notebook using nbformat
        with open(notebook_file_path, 'r', encoding='utf-8') as f:
            notebook_content = nbformat.read(f, as_version=4)
        cell_count = 1
        # Loop through each cell in the book and take out the source code
        code_sources = ""
        for cell in notebook_content.cells:
            if cell.cell_type == 'code':
                source = cell['source']
                
              
                cell_pre = f"<CODE_CELL_{cell_count}>\n"
                cell_finish = f"\n<END_CODE_CELL_{cell_count}>\n\n"
                cell_count += 1
                
                # Append cell identifier and cleaned source code
                code_sources += cell_pre + source + cell_finish
            else:
                source = cell['source']
                cell_pre = f"<MARKDOWN_CELL_{cell_count}>"
                cell_finish = f"\n<END_MARKDOWN_CELL_{cell_count}>\n\n"
                cell_count += 1
