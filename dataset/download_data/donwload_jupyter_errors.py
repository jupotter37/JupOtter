import os
import re
from datasets import load_dataset

# set save folder
save_folder = r''


# regex to catch any character not allowed in Windows filenames
invalid = r'[<>:"/\\|?*]'

def sanitize(name):
    # remove characters causing issues with filenames
    return re.sub(invalid, '', name)

def parse_author_repo(url: str):
    parts = url.split('/')
    return parts[3], parts[4]

total_saved = 0
for split_name in ["train", "test"]: #get split
    print(f"Processing split: {split_name}")
    ds = load_dataset(
        "JetBrains-Research/jupyter-errors-dataset",
        split=split_name,   
        streaming=True,
    )

    for notebook in ds:
        github_url = notebook["file_link"]
        content = notebook["content"]
        notebook_rel = notebook["path"] # used to get file name

        author, repo = parse_author_repo(github_url)
        raw_fname = os.path.basename(notebook_rel)
        # sanitize each component
        author_s = sanitize(author)
        repo_s = sanitize(repo)
        fname_s = sanitize(raw_fname)
        # ensure is ipynb 
        if not fname_s.lower().endswith(".ipynb"):
            fname_s += ".ipynb"
        # construct filename we can trace later and remove duplicates
        save_name = f"{author_s}_{repo_s}_{fname_s}"
        save_path = os.path.join(save_folder, save_name)

        # write
        with open(save_path, "w", encoding="utf-8") as fh:
            fh.write(content) # files with duplicate names will be overwritten, removes duplicates. 
                              # Files can be given unique numbered names (e.g file1, file2, file3) to preserve all notebooks.

        total_saved += 1
        if total_saved % 100 == 0:
            print(f"Saved {total_saved} Python notebooks so far")


