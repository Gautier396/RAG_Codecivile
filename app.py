from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

CHROMA_PATH = "chroma_db"

def main():
    print("⚙️ Initialisation du système RAG...")
    
    # 1. Modèle d'Embedding (doit être le même que pour l'ingestion)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # 2. Chargement de la base vectorielle ChromaDB
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # 3. Modèle LLM (Qwen 2.5) via Ollama
    llm = ChatOllama(model="qwen2.5", temperature=0)
    
    # 4. Prompt RAG
    template = """Tu es un assistant juridique expert en droit français. Utilise le contexte suivant tiré du Code civil pour répondre à la question.
    Si la réponse ne se trouve pas dans le contexte, dis simplement que tu ne sais pas, n'invente rien.
    Cite toujours précisément le ou les numéros des articles sur lesquels tu te bases pour formuler ta réponse.

    Contexte : {context}

    Question : {question}

    Réponse :"""
    prompt = PromptTemplate.from_template(template)
    
    # 5. Pipeline RAG avec LangChain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 6. Boucle de chat interactive
    print("\n✅ Prêt ! Posez vos questions sur le Code civil (tapez 'quit' pour quitter).")
    while True:
        user_query = input("\n⚖️ Vous : ")
        if user_query.lower() in ['quit', 'exit', 'quitter']:
            break
        
        print("🤖 Qwen : ", end="", flush=True)
        for chunk in rag_chain.stream(user_query):
            print(chunk, end="", flush=True)
        print()

if __name__ == "__main__":
    main()