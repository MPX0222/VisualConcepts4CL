from openai import OpenAI
from typing import List, Dict
import os
import json


class DescriptionGenerator:
    def __init__(self):
        self.client = OpenAI(
            api_key="sk-MMAgrrbNFTDIyqgw1cDf1a6189C54fBdA5Cb12C87602431b",
            base_url="https://vip.yi-zhan.top/v1"
        )
        
    def generate_descriptions(self, categories: List[str], prompt_type="default") -> Dict[str, List[str]]:
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Build batch prompts
                prompt = self._build_batch_prompt(categories, prompt_type)
                
                # Call OpenAI API
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini", 
                    messages=[
                        {"role": "system", "content": "You are a visual concept generator for CLIP-based image classification. Your task is to generate concise visual concepts that maximize the similarity between image and text embeddings."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                
                # Parse response
                generated_text = response.choices[0].message.content
                result = self._parse_json_response(generated_text)
                
                # Validate if result is dictionary and contains all categories
                if isinstance(result, dict) and all(cat in result for cat in categories):
                    return result
                    
            except Exception as e:
                print(f"Attempt {retry_count + 1} failed: {str(e)}")
                retry_count += 1
                
        # If all retries fail, return a basic dictionary structure
        return {category: ["basic visual feature"] * 5 for category in categories}
    
    def _build_batch_prompt(self, categories: List[str], prompt_type: str) -> str:
        if prompt_type == "medical":
            return f"""You are a visual concept generator for CLIP-based image classification. Your task is to generate concise visual concepts that maximize the similarity between image and text embeddings. The images are dermatological images captured under the dermatoscope.

Inputs:
- Number of concepts per class: 5
- Class names: {categories}

Output format:
Return a JSON object: {{'class_name1':['concept1','concept2','concept3'], 'class_name2':[...], ...}}

Rules for concept generation:
1. Each concept MUST be 3 words long (e.g., "red_breasted_bird", "long_pointed_beak")
2. Focus on distinctive visual features that are unique to each class
3. Use descriptive adjectives + nouns format (e.g., "spotted_wings", "curved_bill")
4. Avoid non-visual features (e.g., sounds, behaviors, habitats)
5. Do not reference other classes or use comparative terms
6. Each concept should be distinct from others within the same class

Attention: each class must have 5 concepts, and the concepts must be distinct from each other.
Please generate the concepts in JSON format."""

        elif prompt_type == "default":
            return f"""You are a visual concept generator for CLIP-based image classification. Your task is to generate concise visual concepts that maximize the similarity between image and text embeddings.

    Inputs:
    - Number of concepts per class: 5
    - Class names: {categories}

    Output format:
    Return a JSON object: {{'class_name1':['concept1','concept2','concept3'], 'class_name2':[...], ...}}

    Rules for concept generation:
    1. Each concept MUST be 3 words long (e.g., "red_breasted_bird", "long_pointed_beak")
    2. Focus on distinctive visual features that are unique to each class
    3. Use descriptive adjectives + nouns format (e.g., "spotted_wings", "curved_bill")
    4. Avoid non-visual features (e.g., sounds, behaviors, habitats)
    5. Do not reference other classes or use comparative terms
    6. Each concept should be distinct from others within the same class

    Attention: each class must have 5 concepts, and the concepts must be distinct from each other.
    Please generate the concepts in JSON format."""

        elif prompt_type == "ct":
            return f"""You are a visual concept generator for CLIP-based image classification. Your task is to generate concise visual concepts that maximize the similarity between image and text embeddings. All images are coronal CT slices of human organs, with each class corresponding to a different organ.


Inputs:
- Number of concepts per class: 5
- Class names: {categories}

Output format:
Return a JSON object: {{'class_name1':['concept1','concept2','concept3'], 'class_name2':[...], ...}}

Rules for concept generation:
1. Each concept MUST be 3 words long (e.g., "red_breasted_bird", "long_pointed_beak")
2. Focus on distinctive visual features that are unique to each class
3. Use descriptive adjectives + nouns format (e.g., "spotted_wings", "curved_bill")
4. Avoid non-visual features (e.g., sounds, behaviors, habitats)
5. Do not reference other classes or use comparative terms
6. Each concept should be distinct from others within the same class

Attention: each class must have 5 concepts, and the concepts must be distinct from each other.
Please generate the concepts in JSON format."""
    
    def _parse_json_response(self, text: str) -> Dict[str, List[str]]:
        try:
            # Try to parse JSON directly
            result = json.loads(text)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON part
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                except json.JSONDecodeError:
                    raise ValueError("Failed to parse JSON response from API")
            else:
                raise ValueError("No valid JSON found in API response")
        
        # 验证每个键的值是否为列表且长度为3
        for key, value in result.items():
            if not isinstance(value, list):
                raise ValueError(f"Value for key '{key}' is not a list")
            if len(value) != 5:
                raise ValueError(f"List for key '{key}' does not have exactly 5 elements")
        
        return result