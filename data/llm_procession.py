import os
import json
from langchain_ollama import OllamaLLM

MODEL = "llama3"  # LLaMA 3 model adı
model = OllamaLLM(model=MODEL)

# JSON dosyalarının bulunduğu ana klasör
json_folder_path = r"D:\\Yeni klasör\\Tubitak-E-Commerce-RAG\\data\\json_output"
processed_folder_path = r"D:\\Yeni klasör\\Tubitak-E-Commerce-RAG\\data\\processed_json"  # İşlenmiş JSON'lar için hedef klasör
os.makedirs(processed_folder_path, exist_ok=True)

# Dosyaları işle
for folder in ["1", "2", "3"]:  # Klasörleri sırayla al
    folder_path = os.path.join(json_folder_path, folder)
    processed_folder = os.path.join(processed_folder_path, folder)
    os.makedirs(processed_folder, exist_ok=True)
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            
            # JSON dosyasını yükle
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            review_content = data.get("content", "")  # İçeriği al
            
            # Model ile içeriği düzenle
            prompt = f"These are the reviews of the laptops. I want you to refine them to ensure proper English writing and grammatical rules, so that when they are input into another LLM, the LLM can make better inferences.If the sentences are already correct and adhere to language rules, do not modify them; leave them as they are. Just provide the corrected versions of the reviews without any additional comments.: {review_content}"
            response = model.invoke(prompt)
            
            # Düzenlenmiş içeriği güncelle
            data["content"] = response
            
            # İşlenmiş JSON dosyasını kaydet
            processed_file_path = os.path.join(processed_folder, file_name)
            with open(processed_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

print("Düzenlenmiş içerikler kaydedildi.")
