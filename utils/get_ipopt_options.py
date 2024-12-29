"""

This script retrieves the list of all options from the Ipopt documentation and deduces
their types. Using that information, it generates code for the `options.py` module.

"""

import base64

import requests

# GitHub repository information
repo_owner = "coin-or"
repo_name = "ipopt"
file_path = "doc/options.dox"

# API endpoint to retrieve the file contents
api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"

# Send GET request to retrieve the file contents
response = requests.get(api_url)
data = response.json()

# Extract the content from the response
if "content" not in data:
    raise RuntimeError("File not found.")
file_content = data["content"]
decoded_content = base64.b64decode(file_content).decode("utf-8")
pieces = decoded_content.split(r"\anchor")

code = []
for piece in pieces[1:]:
    option = piece.split("\n")[0][5:]
    if "real option" in piece:
        type_ = "float"
    elif "integer option" in piece:
        type_ = "int"
    elif "string option" in piece:
        type_ = "str"
    else:
        print(piece)
        raise Warning("Unknown type.")
    code.append(f"    {option}: {type_}")

# Add the suppress banner option which is not in the documentation
code.append(f"    sb: str")

code.sort()
for line in code:
    print(line)
