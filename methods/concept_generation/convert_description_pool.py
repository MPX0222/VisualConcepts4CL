import json
from typing import Dict, List, Set, Tuple
import os

class DescriptionPoolConverter:
    def __init__(self):
        self.unique_descriptions = []  # Store all unique descriptions
        self.description_to_index = {}  # Mapping from description to index
        self.converted_pool = {}  # Converted description pool (using indices)
    
    def load_original_pool(self, filename: str):
        """Load original description pool data"""
        with open(filename, 'r', encoding='utf-8') as f:
            self.original_pool = json.load(f)
    
    def extract_unique_descriptions(self):
        """Extract all unique descriptions"""
        # Use set to ensure descriptions are unique
        unique_desc_set = set()
        for category, descriptions in self.original_pool.items():
            unique_desc_set.update(descriptions)
        
        # Convert to list and maintain order
        self.unique_descriptions = list(unique_desc_set)
        
        # Create mapping from description to index
        self.description_to_index = {
            desc: idx for idx, desc in enumerate(self.unique_descriptions)
        }
    
    def convert_pool_to_indices(self):
        """Convert description pool to use indices"""
        for category, descriptions in self.original_pool.items():
            # Convert each description to its corresponding index
            indices = [self.description_to_index[desc] for desc in descriptions]
            self.converted_pool[category] = indices
    
    def save_unique_descriptions(self, filename: str):
        """Save unique description list to file"""
        print(len(self.unique_descriptions))
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(str(self.unique_descriptions))
    
    def save_converted_pool(self, filename: str):
        """Save converted description pool to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.converted_pool, f, ensure_ascii=False, indent=2)
    
    def convert(self, input_pool_file: str, output_descriptions_file: str, output_pool_file: str):
        """Execute complete conversion process"""
        print("Starting description pool conversion...")
        
        # Load original description pool
        print("Loading original description pool...")
        self.load_original_pool(input_pool_file)
        
        # Extract unique descriptions
        print("Extracting unique descriptions...")
        self.extract_unique_descriptions()
        print(f"Found {len(self.unique_descriptions)} unique descriptions")
        
        # Convert description pool
        print("Converting description pool to index form...")
        self.convert_pool_to_indices()
        
        # Save results
        print("Saving unique description list...")
        self.save_unique_descriptions(output_descriptions_file)
        
        print("Saving converted description pool...")
        self.save_converted_pool(output_pool_file)
        
        print("Conversion complete!")
        print(f"Unique descriptions saved to: {output_descriptions_file}")
        print(f"Converted description pool saved to: {output_pool_file}")

def main():
    # Set file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_pool_file = os.path.join('description_pool', 'description_pool_v4.json')
    output_descriptions_file = os.path.join(current_dir, "unique_descriptions.txt")
    output_pool_file = os.path.join(current_dir, "description_pool_indices.json")
    
    # Create converter and execute conversion
    converter = DescriptionPoolConverter()
    converter.convert(input_pool_file, output_descriptions_file, output_pool_file)

if __name__ == "__main__":
    main() 