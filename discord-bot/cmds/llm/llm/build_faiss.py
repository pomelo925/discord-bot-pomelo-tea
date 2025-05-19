import os
import yaml
import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

### === 1. 載入 YAML 設定 === ###
with open("./../config/build_faiss.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

dataset_class = config["dataset_class"]
raw_data_name = config["raw_data_name"]
embedding_model = config["embedding_model"]

### === 2. 讀取 Excel === ###
excel_path = f"./../database/{dataset_class}/{raw_data_name}"
df = pd.read_excel(excel_path).fillna("NULL")

### === 3. 轉換成 chunks 字串 === ###
documents = []
for _, row in df.iterrows():
    content = ", ".join([f"{col}: {row[col]}" for col in df.columns])
    documents.append(Document(page_content=content))

# 儲存純文字版本（可選）
txt_output_path = f"./../database/{dataset_class}/{raw_data_name}.txt"
with open(txt_output_path, "w", encoding="utf-8") as f:
    f.write("\n".join([doc.page_content for doc in documents]))

### === 4. 建立 Embedding Model === ###
embedding = HuggingFaceEmbeddings(model_name=embedding_model)

### === 5. 建立 FAISS Index 並儲存 === ###
output_dir = f"./../database/{dataset_class}"
os.makedirs(output_dir, exist_ok=True)
db = FAISS.from_documents(documents, embedding)
db.save_local(output_dir)

print(f"✅ FAISS index 已儲存於：{output_dir}/index.faiss + index.pkl")