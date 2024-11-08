from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub

def get_answer_or_summary(query, vectorstore):
    # Retrieve the QA chat prompt template from the LangChain hub
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Initialize a document combination chain with a Chat model
    combine_docs_chain = create_stuff_documents_chain(
        ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0),
        retrieval_qa_chat_prompt
    )

    # Create the retrieval chain using the vector store as a retriever
    retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

    # Run the retrieval chain with the query and get the result
    result = retrieval_chain.invoke({"input": query})

    # Return the generated answer or summary
    return result["answer"]
