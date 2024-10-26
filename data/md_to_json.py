import json
import os

def md_to_json(md_folder_path, json_folder_path):
    for root, _, files in os.walk(md_folder_path):
        for md_filename in files:
            if md_filename.endswith('.md'):
                md_file_path = os.path.join(root, md_filename)
                
                with open(md_file_path, 'r', encoding='utf-8') as md_file:
                    lines = md_file.readlines()
                
                data = {}
                content = []
                capture_content = False  # Başlangıçta içeriği yakalamayı kapalı yapıyoruz

                for line in lines:
                    line = line.strip()
                    
                    # Başlıkları ve içerikleri ayır
                    if line.startswith("### Review's Title:"):
                        data["title"] = line.split(":", 1)[1].strip()
                    elif line.startswith("**Review's Content**:"):
                        capture_content = True  # İçeriği yakalamaya başla
                    elif capture_content and not line.startswith("**"):
                        content.append(line)
                    elif line.startswith("**Review's Likes**:"):
                        data["likes"] = int(line.split(":", 1)[1].strip())
                    elif line.startswith("**Review's Dislikes**:"):
                        data["dislikes"] = int(line.split(":", 1)[1].strip())
                
                data["content"] = " ".join(content).strip()
                
                # JSON dosyasını kaydet
                json_filename = md_filename.replace('.md', '.json')
                json_file_path = os.path.join(json_folder_path, json_filename)
                os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
                
                with open(json_file_path, 'w', encoding='utf-8') as json_file:
                    json.dump(data, json_file, ensure_ascii=False, indent=4)

# Kullanım
md_folder_path = r"D:\\Yeni klasör\\Tubitak-E-Commerce-RAG\\data\\markdown\\laptops\\3"
json_folder_path = r"D:\Yeni klasör\Tubitak-E-Commerce-RAG\data\json_output\\3"
os.makedirs(json_folder_path, exist_ok=True)

md_to_json(md_folder_path, json_folder_path)
