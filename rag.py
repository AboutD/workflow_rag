import markdown
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.vectorstores import FAISS
# 知识库创建并保存
class createragbase:
    def __init__(self, document_path, embeddings, document_type='markdown'):
        '''
        document_path：文档路径
        document_type：文档类型
        embeddings：langchain调用的embedding模型
        '''
        self.document_path = document_path
        self.document_type = document_type
        self.embeddings = embeddings
        
    def parse_markdown(self):
        with open(self.document_path, 'r', encoding='utf-8') as f:
            text = f.read()
        html = markdown.markdown(text)
        soup = BeautifulSoup(html, features="html.parser")
        sections = soup.find_all(['h1', 'p'])
        documents = []
        current_content = ""
        current_title = None
        for section in sections:
            if section.name == 'h1':
                if current_title is not None:
                    documents.append(Document(page_content=f"{current_title}\n{current_content}", metadata={"title": current_title}))
                current_title = section.get_text()
                current_content = ""
            else:
                current_content += section.get_text() + "\n"
        if current_title is not None:
            documents.append(Document(page_content=f"{current_title}\n{current_content}", metadata={"title": current_title}))
        return documents
    
    def fit(self):
        if self.document_type == 'markdown':
            document_chunks = self.parse_markdown()
            db = FAISS.from_documents(document_chunks, self.embeddings)
            return db
        else:
            raise ValueError("Unsupported document type")
    
    def fit_save(self, file_path="faiss_index"):
        if self.document_type == 'markdown':
            document_chunks = self.parse_markdown()
            db = FAISS.from_documents(document_chunks, self.embeddings)
            # 保存到本地
            db.save_local(file_path)
            print(f'本地知识库已保存到：{file_path}')
        else:
            raise ValueError("Unsupported document type")
# 工作流知识库单元
class workflowragcell:
    def __init__(self, embeddings, db_path="faiss_index"):
        '''
        db_path：本地知识库保存路径
        embeddings：langchain调用的embedding模型，注意要和embedding知识库的模型相同
        '''
        self.db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    
    def search(self, query, k):
        result = self.db.similarity_search(query, k=k)
        return result
# 创建并保存知识库示例
from langchain_community.embeddings import BaichuanTextEmbeddings

# 定义embedding模型
bcembeddings = BaichuanTextEmbeddings(baichuan_api_key="sk-d2a8e9dbca98d4a1567fd9608b2966a2")

# 定义文档路径
markdown_path = "finance.md"

# 创建并保存知识库到 rag_local 文件夹下
createragbase(markdown_path, bcembeddings, 'markdown').fit_save('rag_local/faiss_index')

# 工作流知识库单元调用示例
from langchain_community.embeddings import BaichuanTextEmbeddings
from langchain.vectorstores import FAISS

# 定义embedding模型
bcembeddings = BaichuanTextEmbeddings(baichuan_api_key="sk-d2a8e9dbca98d4a1567fd9608b2966a2")

# 定义知识库路径
rag_path = r"C:\Python\Own_Agent\rag_local\faiss_index"

# 创建知识库并查询
rag1 = workflowragcell(bcembeddings, rag_path)
print(rag1.search('偿债能力相关指标', k=1)[0].page_content)