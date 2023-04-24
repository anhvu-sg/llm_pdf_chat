from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ChatVectorDBChain
from langchain.chat_models import ChatOpenAI


pdf_path = "./rule_of_life.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

embeddings = OpenAIEmbeddings()
vector_db = Chroma.from_documents(
    pages, embedding=embeddings, persist_directory="."
)

vector_db.persist()

retriever = ChatVectorDBChain.from_llm(
    ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
    vector_db,
    return_source_documents=True
)

# query = "Summary of the pdf"
# query = "Take a step back and consider the following questions?"

print("Please input your question, such as: 'Summary of the pdf' ")
while input != "quit()":
    query = input()
    result = retriever({
        "question": query,
        "chat_history": ""
    })
    print("\nYour Question: {query}".format(query=query))
    print("Answer:")
    print(result["answer"])
    print("\n\nYour next question:")