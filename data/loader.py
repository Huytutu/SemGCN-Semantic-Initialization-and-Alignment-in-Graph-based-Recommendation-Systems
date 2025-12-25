# =============================================================================
# File I/O Utilities
# =============================================================================
# Handles loading and saving data files:
# - Training/test datasets (user-item interactions)
# - Results and logs
# =============================================================================

import os.path
from os import remove
from re import split
import csv


class FileIO(object):
    """Static utility class for file input/output operations."""
    
    def __init__(self):
        pass

    @staticmethod
    def write_file(dir, file, content, op='w'):
        """Write content to a file.
        
        Args:
            dir (str): Directory path (created if doesn't exist)
            file (str): Filename
            content (list): Lines to write
            op (str): File mode ('w'=write, 'a'=append)
        """
        # Create directory if it doesn't exist
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        # Write content line by line
        with open(dir + file, op) as f:
            f.writelines(content)

    @staticmethod
    def delete_file(file_path):
        """Delete a file if it exists.
        
        Args:
            file_path (str): Path to file to delete
        """
        if os.path.exists(file_path):
            remove(file_path)

    @staticmethod
    def load_data_set(file, rec_type):
        """Load training or test dataset from CSV.

        Expected columns: user_id,item_id,rating (or user_id,parent_asin,rating).
        Automatically skips a header row and blank/invalid lines.
        """
        data = []
        if rec_type == 'graph':
            with open(file, newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or len(row) < 3:
                        continue
                    user_id = row[0].strip()
                    item_id = row[1].strip()
                    weight_str = row[2].strip()
                    # Skip header if present
                    if (
                        user_id.lower() in ('user_id', 'user') or
                        item_id.lower() in ('item_id', 'item', 'parent_asin') or
                        weight_str.lower() in ('rating', 'weight')
                    ):
                        continue
                    try:
                        weight = float(weight_str)
                    except ValueError:
                        continue
                    data.append([user_id, item_id, weight])
        return data

    @staticmethod
    def load_user_list(file):
        """Load a list of user IDs from file.

        Supports CSV with a header or plain text (one ID per line).
        """
        user_list = []
        print('loading user List...')
        with open(file, newline='') as f:
            reader = csv.reader(f)
            header_checked = False
            user_idx = 0
            for row in reader:
                if not row:
                    continue
                if not header_checked:
                    # Detect header and find 'user_id' column
                    lower = [c.strip().lower() for c in row]
                    if 'user_id' in lower or 'user' in lower:
                        user_idx = lower.index('user_id') if 'user_id' in lower else lower.index('user')
                        header_checked = True
                        continue  # skip header row
                    header_checked = True  # no header detected
                if len(row) <= user_idx:
                    continue
                user_list.append(row[user_idx].strip())
        return user_list
