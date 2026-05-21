from langchain_text_splitters import RecursiveCharacterTextSplitter
import os 

def chunk_wiki_files(directory='wiki'):
    documents = []

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    'title': filename.replace('.txt', ''),
                    'content': content
                })


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []
    for doc in documents:
        split_text = text_splitter.split_text(doc['content'])
        for chunk in split_text:
            chunks.append({
                'text': chunk,
                'source': doc['title']
            })
    return chunks 
