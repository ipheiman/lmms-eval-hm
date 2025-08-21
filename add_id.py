import json

# Path to your input and output JSONL files
input_file = "input.jsonl"
output_file = "output_with_id.jsonl"

# Read the original JSONL file
with open(input_file, "r") as f:
    lines = f.readlines()

# Add an "id" field to each entry
new_lines = []
for idx, line in enumerate(lines):
    data = json.loads(line)
    data["id"] = str(idx)  # use string id like in your example
    new_lines.append(json.dumps(data))

# Write back to a new JSONL file
with open(output_file, "w") as f:
    for line in new_lines:
        f.write(line + "\n")

print(f"Saved new JSONL with IDs to {output_file}")
