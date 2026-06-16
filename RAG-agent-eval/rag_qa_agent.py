from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGAgent:

    def __init__(
            self,
            document_paths: list = None,
            embedding_model=None,
            chunk_size: int = 500,
            chunk_overlap: int = 50,
            vector_store_class = FAISS,
            k: int = 2
    ):
        self.document_paths = document_paths
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store_class = vector_store_class
        self.k = k
        self.vector_store = self._load_vector_store()


    def _load_vector_store(self):
        documents = []
        for document_path in self.document_paths:
            with open(document_path, "r", encoding="utf-8") as file:
                raw_text = file.read()
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size = self.chunk_size,
                chunk_overlap = self.chunk_overlap 
            )

            documents.extend(splitter.create_documents([raw_text]))

        return self.vector_store_class.from_documents(documents, self.embedding_model)
        

    def retrieve(self, query: str):
        docs = self.vector_store.similarity_search(query, k = self.k)
        context = [ doc.page_content for doc in docs]
        return context
    
    def generate(self, query: str, retrieved_docs: list):
        # retrieved_docs = self.retrieve("How many blood tests can you perform and how much blood do you need?")
        context = "\n".join(retrieved_docs)
        model = OpenAI(temperature=0)
        prompt = ("Answer the query using the context below.\n\nContext:\n{context}\n\nQuery:\n{query}"
            "Only use information from the context. If nothing relevant is found, respond with: 'No relevant information available.'"
        )
        prompt = prompt.format(context = context, query = query)
        return model.invoke(prompt)
    
    def answer(self, query:str) -> tuple[str, list[str]]:
        retrieved_docs = retreiver.retrieve(query)
        generated_answer = retreiver.generate(query, retrieved_docs)
        return generated_answer, retrieved_docs


if __name__ == "__main__":
    document_paths = ["./RAG-agent-eval/dataset/theranos_legacy.txt"]
    query = "How many blood tests can you perform and how much blood do you need?"

    retreiver = RAGAgent(document_paths)
    answer, retrieved_docs = retreiver.answer(query)
