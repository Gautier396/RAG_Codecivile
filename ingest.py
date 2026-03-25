import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from ollama._types import ResponseError

CHROMA_PATH = "chroma_db"
DATA_DIR = "data"

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 2. Ingestion
    print("1. Chargement des données depuis le dossier 'data'...")
    documents = []
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        if filename.endswith(".txt"):
            documents.extend(TextLoader(filepath, encoding="utf-8").load())
            print(f"   - Fichier texte chargé : {filename}")
        elif filename.endswith(".pdf"):
            documents.extend(PyPDFLoader(filepath).load())
            print(f"   - Fichier PDF chargé : {filename}")
            
    if not documents:
        print("❌ Aucun fichier .txt ou .pdf n'a été trouvé dans le dossier 'data'. Ajoutez votre fichier du Code civil et relancez.")
        return

    # 3. Découpage (Chunking)
    print("2. Découpage (Chunking)...")
    # Utilisation d'une expression régulière pour couper en priorité au début de chaque "Article "
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[r"\n(?=Article )", r"\n\n", r"\n", " ", ""],
        is_separator_regex=True,
        chunk_size=1500,  # Réduit pour éviter de dépasser la limite de contexte du modèle d'embedding
        chunk_overlap=150 # Adapté à la nouvelle taille
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   -> {len(chunks)} fragments (chunks) créés.")

    # 4. Base de données vectorielle & Embedding
    print("3. Création des embeddings et stockage dans ChromaDB...")
    try:
        # Utilisation de nomic-embed-text via Ollama pour l'embedding
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
        print(f"✅ Base de données vectorielle initialisée et sauvegardée dans '{CHROMA_PATH}'.")
    except ResponseError as e:
        print("\n❌ Erreur de connexion à Ollama.")
        print("Veuillez vérifier deux choses :")
        print("1. L'application Ollama est-elle bien en cours d'exécution sur votre machine ?")
        print(f"2. Le modèle d'embedding a-t-il été téléchargé ? Essayez : 'ollama pull nomic-embed-text'")
        print(f"\nErreur originale : {e}")


if __name__ == "__main__":
    main()