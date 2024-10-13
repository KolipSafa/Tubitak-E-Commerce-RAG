import json
import random
import os

class Utils:
    @staticmethod
    def check_dir(directory) -> None:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    @staticmethod
    def get_random_number() -> int:
        return random.randint(1, 100)
    
    @staticmethod
    def save_docs_with_scores(docs_with_scores, file_path):
        data = []
        for doc, score in docs_with_scores:
            if hasattr(score, 'item'):
                score = score.item()
            else:
                score = float(score)

            data.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': score
            })
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    @staticmethod
    def save_llm_result(full_prompt: str, response: str, file_path):
        """Save the prompt and LLM response to a JSON file"""

        output_data = {
            "prompt": full_prompt,
            "response": response
        }

        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(output_data, json_file, ensure_ascii=False, indent=4)
