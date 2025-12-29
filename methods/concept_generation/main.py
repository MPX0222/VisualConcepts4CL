import json
import os
from typing import List, Dict
from description_generator import DescriptionGenerator
from description_pool import DescriptionPool

import nltk

nltk.download('stopwords')
nltk.download('punkt_tab')

class ContinuousLearningSystem:
    def __init__(self, data_info: dict):
        self.category_file = data_info["path"]
        self.batch_size = data_info["batch_size"]
        self.need_type = data_info["need_type"]
        self.need_description = data_info["need_description"]

        self.description_generator = DescriptionGenerator()
        self.description_pool = DescriptionPool()
        
    def load_categories(self) -> List[str]:
        with open(self.category_file, 'r') as f:
            categories = eval(f.read())
        return categories
    
    def process_batch(self, categories: List[str], iteration: int, prompt_type: str):
        # 为当前批次生成描述

        new_descriptions = self.description_generator.generate_descriptions(categories, prompt_type)
        
        # 更新描述池
        self.description_pool.update(new_descriptions)
        
        # 保存当前版本的描述池
        output_folder = "description_pool"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_file = f"description_pool_v{iteration}.json"
        self.description_pool.save(os.path.join(output_folder, output_file))
        
        print(f"完成第 {iteration} 轮处理，生成了 {len(new_descriptions)} 条新描述")
        print(f"当前描述池大小: {len(self.description_pool.pool)} 条描述")
    
    def run(self):
        categories = self.load_categories()
        total_iterations = (len(categories) + self.batch_size - 1) // self.batch_size

        count_index = 0
        
        if self.need_type == "all":
            prompt_type = self.need_description["type"]
            for i in range(total_iterations):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(categories))
                batch_categories = categories[start_idx:end_idx]

                print(f"\n开始处理第 {i+1} 轮，包含类别: {batch_categories}, 描述类型: {prompt_type}")
                self.process_batch(batch_categories, i+1, prompt_type)

        elif self.need_type == "mixed":
            tag_dict = {tag["end_index"]: tag["type"] for tag in self.need_description}
            tag_dict_keys = sorted(list(tag_dict.keys()))
            cur_tag = 0

            for i in range(total_iterations):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(categories))
                batch_categories = categories[start_idx:end_idx]

                if end_idx <= tag_dict_keys[cur_tag]:
                    prompt_type = tag_dict[tag_dict_keys[cur_tag]]
                    print(f"\n开始处理第 {i+1} 轮，包含类别: {batch_categories}, 描述类型: {prompt_type}`")
                    self.process_batch(batch_categories, i+1, prompt_type)
                else:
                    cur_tag += 1
                    prompt_type = tag_dict[tag_dict_keys[cur_tag]]
                    print(f"\n开始处理第 {i+1} 轮，包含类别: {batch_categories}, 描述类型: {prompt_type}")
                    self.process_batch(batch_categories, i+1, prompt_type)


if __name__ == "__main__":
    # data_info = {
    #     "path": "datasets/skin40.txt",
    #     "need_type": "mixed",
    #     "need_description": [
    #         {
    #             "type": "medical",
    #             "start_index": 0,
    #             "end_index": 6
    #         },
    #         {
    #             "type": "ct",
    #             "start_index": 6,
    #             "end_index": 18
    #         }
    #     ],
    #     "batch_size": 2
    # }

    data_info = {
        "path": "datasets/skin8.txt",
        "need_type": "all",
        "need_description": {
            "type": "default"
        },
        "batch_size": 2
    }

    system = ContinuousLearningSystem(data_info)
    system.run() 