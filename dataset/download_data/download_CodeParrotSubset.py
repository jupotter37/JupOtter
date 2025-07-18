import os
import re
from datasets import load_dataset
import nbformat


# 1) where to save
save_folder = r''
os.makedirs(save_folder, exist_ok=True)

# regex to catch any character not allowed in Windows filenames
invalid = r'[<>:"/\\|?*]'

def sanitize(name):
    # remove characters causing issues with filenames
    return re.sub(invalid, '', name)

def parse_author_repo(url):
    parts = url.split('/')
    return parts[0] + "_" + parts[1]


def parse_notebook_name(url):
    parts = url.split('/')
    return parts[-1]

total_saved_attempt = 0

for split_name in ["train", "test"]:
    print(f"Processing split: {split_name}")
    ds = load_dataset(
        "codeparrot/github-jupyter",
        split=split_name,   
        streaming=True,
    )

    for notebook in ds:
        repo_auth = sanitize(parse_author_repo(notebook["repo_name"]))
        notebookName = sanitize(parse_notebook_name(notebook["path"]))
        content = notebook["content"]

        try:
            nb = nbformat.reads(content, as_version=4)
        except Exception:
            continue  # skip invalid notebooks

        lang = nb.metadata.get("language_info", {}).get("name", "").lower()
        if not lang:
            lang = nb.metadata.get("kernelspec", {}).get("language", "").lower()
        if lang != "python":
            continue # skip invalid notebooks


        if not notebookName.lower().endswith(".ipynb"):
            notebookName += ".ipynb"

        save_name = f"{repo_auth}_{notebookName}"
        save_path = os.path.join(save_folder, save_name)

        # write
        with open(save_path, "w", encoding="utf-8") as fh:
            fh.write(content) # files with duplicate names will be overwritten, removes duplicates
                            # Files can be given unique numbered names (e.g file1, file2, file3) to preserve all notebooks.


        total_saved_attempt += 1
        if total_saved_attempt % 100 == 0:
            print(f"Saved {total_saved_attempt} Python notebooks so far")
        if total_saved_attempt == 5000:
            break
    if total_saved_attempt == 5000:
            break


print(f"Total saved notebooks: {total_saved_attempt}")