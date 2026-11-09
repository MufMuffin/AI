import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

file_names = ['document1.txt', 'document2.txt', 'document3.txt', 'document4.txt']

chunks = []
chunk_id = 1

for file_name in file_names:
    file_path = os.path.join(BASE_DIR, file_name)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
            raw_chunks = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            for paragraph in raw_chunks:
                chunks.append({
                    "id": f"chunk_{chunk_id:02d}",
                    "content": paragraph,
                    "source": file_name
                })
                chunk_id += 1

output_dir = os.path.join(BASE_DIR, "chunks_output")
os.makedirs(output_dir, exist_ok=True)

for item in chunks:
    file_name = f"{item['id']}.txt"
    file_path = os.path.join(output_dir, file_name)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"Source: {item['source']}\n")
        f.write(f"ID: {item['id']}\n")
        f.write("-" * 40 + "\n\n")
        f.write(item['content'])

print(f"Успішно створено та збережено {len(chunks)} файлів у папку:\n{output_dir}")