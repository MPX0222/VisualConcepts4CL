import json
from typing import Dict, List, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

class DescriptionPool:
    def __init__(self):
        self.pool = {}  # 存储所有描述
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # 使用1-gram和2-gram
        self.description_vectors = None
        self.all_descriptions = []
        self.description_to_category = {}  # 记录每个描述属于哪个类别
        self.category_to_descriptions = defaultdict(set)  # 记录每个类别包含哪些描述
        self.target_descriptions_per_category = 5  # 每个类别目标描述数量
        self.stop_words = set(stopwords.words('english'))
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for similarity comparison"""
        # 转换为小写
        text = text.lower()
        # 替换下划线为空格
        text = text.replace('_', ' ')
        # 分词
        tokens = word_tokenize(text)
        # 移除停用词
        tokens = [t for t in tokens if t not in self.stop_words]
        return ' '.join(tokens)
    
    def _calculate_similarity_scores(self, desc1: str, desc2: str) -> Dict[str, float]:
        """Calculate multiple similarity scores between two descriptions"""
        # 预处理文本
        desc1_processed = self._preprocess_text(desc1)
        desc2_processed = self._preprocess_text(desc2)
        
        # 计算TF-IDF相似度
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform([desc1_processed, desc2_processed])
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            tfidf_similarity = 0.0
        
        # 计算词重叠率
        words1 = set(desc1_processed.split())
        words2 = set(desc2_processed.split())
        if not words1 or not words2:
            word_overlap = 0.0
        else:
            word_overlap = len(words1.intersection(words2)) / max(len(words1), len(words2))
        
        # 计算词序相似度
        def get_word_sequence(text):
            return [w for w in text.split() if w not in self.stop_words]
        
        seq1 = get_word_sequence(desc1_processed)
        seq2 = get_word_sequence(desc2_processed)
        
        # 计算最长公共子序列
        def lcs(seq1, seq2):
            if not seq1 or not seq2:
                return 0
            if seq1[0] == seq2[0]:
                return 1 + lcs(seq1[1:], seq2[1:])
            return max(lcs(seq1[1:], seq2), lcs(seq1, seq2[1:]))
        
        lcs_length = lcs(seq1, seq2)
        sequence_similarity = lcs_length / max(len(seq1), len(seq2)) if seq1 and seq2 else 0.0
        
        return {
            'tfidf': tfidf_similarity,
            'word_overlap': word_overlap,
            'sequence': sequence_similarity
        }
    
    def _calculate_overall_similarity(self, scores: Dict[str, float], is_same_category: bool = False) -> Tuple[float, bool]:
        """Calculate overall similarity score and determine if it's conflicting
        
        Args:
            scores: Dictionary of individual similarity scores
            is_same_category: Whether the comparison is within the same category
            
        Returns:
            Tuple of (overall_similarity_score, is_conflicting)
        """
        # # 相似度权重
        # weights = {
        #     'tfidf': 0.5,
        #     'word_overlap': 0.3,
        #     'sequence': 0.2
        # }
        
        # # 相似度阈值
        # thresholds = {
        #     'tfidf': 0.7,      # TF-IDF相似度阈值（从0.7提高到0.85）
        #     'word_overlap': 0.6,  # 词重叠率阈值（从0.6提高到0.75）
        #     'sequence': 0.5    # 词序相似度阈值（从0.5提高到0.65）
        # }

        threshold = 0.5
        
        # # 计算加权相似度
        # overall_similarity = sum(scores[metric] * weights[metric] for metric in scores)
        
        # 判断是否冲突
        threshold_multiplier = 1.0 if is_same_category else 1.2

        is_conflicting = scores['tfidf'] * threshold_multiplier > threshold
        overall_similarity = scores['tfidf'] * threshold_multiplier

        # is_conflicting = any(
        #     scores[metric] > thresholds[metric] * threshold_multiplier 
        #     for metric in thresholds
        # )
        
        return overall_similarity, is_conflicting
    
    def _find_most_similar_descriptions(self, desc: str, original_descriptions: List[str], n: int = 3) -> List[Tuple[str, float]]:
        """Find the most similar descriptions from the pool"""
        similarities = []
        for existing_desc in original_descriptions:
            scores = self._calculate_similarity_scores(desc, existing_desc)
            overall_similarity, _ = self._calculate_overall_similarity(scores)
            similarities.append((existing_desc, overall_similarity))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]
    
    def _is_conflicting(self, desc: str, category: str, original_descriptions: List[str]) -> bool:
        """Check if new description conflicts with original descriptions using multiple similarity metrics"""
        if not original_descriptions:
            return False
        
        # 检查与同一类别的描述
        category_descs = self.category_to_descriptions[category]
        for existing_desc in original_descriptions:
            if existing_desc in category_descs:
                scores = self._calculate_similarity_scores(desc, existing_desc)
                _, is_conflicting = self._calculate_overall_similarity(scores, is_same_category=True)
                if is_conflicting:
                    print(f"-- {desc} Conflict detected with category description: {existing_desc}")
                    print(f"Similarity scores: {scores}")
                    return True
        
        # 检查与其他类别的描述
        for existing_desc in original_descriptions:
            if existing_desc not in category_descs:
                scores = self._calculate_similarity_scores(desc, existing_desc)
                _, is_conflicting = self._calculate_overall_similarity(scores, is_same_category=False)
                if is_conflicting:
                    print(f"-- {desc} Conflict detected with other category description: {existing_desc}")
                    print(f"Similarity scores: {scores}")
                    return True
        
        return False
    
    def update(self, new_descriptions: Dict[str, List[str]]):
        # Check description count before update
        print("\nDescription count before update:")
        for category in self.pool:
            count = len(self.category_to_descriptions[category])
            print(f"{category}: {count} descriptions")
            if count != self.target_descriptions_per_category:
                print(f"Warning: {category} does not have {self.target_descriptions_per_category} descriptions")

        # Get all descriptions from the original pool
        original_descriptions = list(self.all_descriptions)
        
        # If pool is empty, add new descriptions directly
        if not original_descriptions:
            self._add_new_descriptions(new_descriptions)
            return
        
        # Update vectors
        self._update_vectors()
        
        # Process each new category
        for category, descs in new_descriptions.items():
            final_descriptions = []
            for desc in descs:
                if not self._is_conflicting(desc, category, original_descriptions):
                    final_descriptions.append(desc)
                else:
                    print(f"Conflicting description: {desc}")
            
            # If new descriptions are insufficient, supplement with most similar existing descriptions
            current_count = len(self.category_to_descriptions[category])
            needed_count = self.target_descriptions_per_category - (current_count + len(final_descriptions))
            
            if needed_count > 0:
                # Use the first new description as reference to find similar existing descriptions
                if descs:
                    reference_desc = descs[0]
                    similar_descs = self._find_most_similar_descriptions(reference_desc, original_descriptions, needed_count)
                    for similar_desc, similarity in similar_descs:
                        if similar_desc not in final_descriptions and similar_desc not in self.category_to_descriptions[category]:
                            final_descriptions.append(similar_desc)
                            print(f"Added similar description: {similar_desc} (similarity: {similarity:.2f})")
            
            # Update description pool
            if final_descriptions:
                self._add_category_descriptions(category, final_descriptions)
        
        # Update vectors
        self._update_vectors()

        # Check and adjust description count after update
        print("\nDescription count after update:")
        for category in self.pool:
            count = len(self.category_to_descriptions[category])
            print(f"{category}: {count} descriptions")
            
            # If description count exceeds target, remove excess descriptions
            if count > self.target_descriptions_per_category:
                excess = count - self.target_descriptions_per_category
                print(f"Warning: {category} has more than {self.target_descriptions_per_category} descriptions, removing {excess}")
                # Keep the newest descriptions, remove the oldest ones
                descriptions_to_keep = list(self.category_to_descriptions[category])[-self.target_descriptions_per_category:]
                self.category_to_descriptions[category] = set(descriptions_to_keep)
                self.pool[category] = descriptions_to_keep
                # Update all_descriptions and description_to_category
                self.all_descriptions = [desc for desc in self.all_descriptions if desc in descriptions_to_keep]
                self.description_to_category = {desc: cat for desc, cat in self.description_to_category.items() if desc in descriptions_to_keep}
            
            # If description count is insufficient, borrow similar descriptions from other categories
            elif count < self.target_descriptions_per_category:
                needed = self.target_descriptions_per_category - count
                print(f"Warning: {category} has less than {self.target_descriptions_per_category} descriptions, adding {needed}")
                # Find similar descriptions from other categories
                other_descriptions = [desc for desc in self.all_descriptions if desc not in self.category_to_descriptions[category]]
                if other_descriptions:
                    reference_desc = list(self.category_to_descriptions[category])[0] if self.category_to_descriptions[category] else None
                    if reference_desc:
                        similar_descs = self._find_most_similar_descriptions(reference_desc, other_descriptions, needed)
                        for similar_desc, similarity in similar_descs:
                            if similar_desc not in self.category_to_descriptions[category]:
                                self._add_category_descriptions(category, [similar_desc])
                                print(f"Borrowed description from other category: {similar_desc} (similarity: {similarity:.2f})")
        
        # Update vectors one final time
        self._update_vectors()
        
        # Print final state
        print("\nFinal description count:")
        for category in self.pool:
            count = len(self.category_to_descriptions[category])
            print(f"{category}: {count} descriptions")
    
    def _add_category_descriptions(self, category: str, descriptions: List[str]):
        """Add descriptions for a category to the pool"""
        if category not in self.pool:
            self.pool[category] = []
        
        for desc in descriptions:
            if desc not in self.pool[category]:
                self.pool[category].append(desc)
                self.all_descriptions.append(desc)
                self.description_to_category[desc] = category
                self.category_to_descriptions[category].add(desc)
    
    def _add_new_descriptions(self, new_descriptions: Dict[str, List[str]]):
        """Add completely new descriptions to the pool"""
        for category, descriptions in new_descriptions.items():
            self._add_category_descriptions(category, descriptions)
    
    def _update_vectors(self):
        """Update TF-IDF vectors"""
        if self.all_descriptions:
            self.description_vectors = self.vectorizer.fit_transform(self.all_descriptions)
    
    def save(self, filename: str):
        """Save description pool to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.pool, f, ensure_ascii=False, indent=2)
    
    def load(self, filename: str):
        """Load description pool from file"""
        with open(filename, 'r', encoding='utf-8') as f:
            self.pool = json.load(f)
            self.all_descriptions = []
            self.description_to_category = {}
            self.category_to_descriptions = defaultdict(set)
            
            for category, descs in self.pool.items():
                for desc in descs:
                    self.all_descriptions.append(desc)
                    self.description_to_category[desc] = category
                    self.category_to_descriptions[category].add(desc)
            
            self._update_vectors()
    
    def get_statistics(self) -> Dict:
        """Get statistics of the description pool"""
        return {
            "total_categories": len(self.pool),
            "total_descriptions": len(self.all_descriptions),
            "descriptions_per_category": {
                category: len(descs) for category, descs in self.pool.items()
            }
        } 