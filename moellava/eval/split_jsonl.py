import json
import os

def split_jsonl(input_file, n):
    """
    Split a JSONL file into n smaller JSONL files.
    
    :param input_file: Path to the input JSONL file.
    :param n: Number of parts to split the file into.
    """
    # Extract the base filename without extension
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    # Ensure the output directory exists
    output_dir = os.path.dirname(input_file)
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Calculate the number of lines per part
    total_lines = len(lines)
    lines_per_part = total_lines // n
    remainder = total_lines % n
    part_file_list = []
    start = 0
    for i in range(n):
        end = start + lines_per_part + (1 if i < remainder else 0)
        part_lines = lines[start:end]
        
        # Write the part to a new file
        part_file = os.path.join(output_dir, f'{base_filename}_part{i+1}.jsonl')
        part_file_list.append(part_file)
        with open(part_file, 'w', encoding='utf-8') as pf:
            pf.writelines(part_lines)
        
        start = end
    return part_file_list
    
    print(f"Split {total_lines} lines into {n} parts in directory '{output_dir}'")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split a JSONL file into multiple parts")
    parser.add_argument('--input_file', type=str, help="Path to the input JSONL file")
    parser.add_argument('--n', type=int, help="Number of parts to split the file into")
    
    args = parser.parse_args()
    
    part_file_list = split_jsonl(args.input_file, args.n)
    for part_file in part_file_list:
        print(part_file)
