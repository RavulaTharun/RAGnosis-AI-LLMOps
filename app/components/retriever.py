from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from app.components.llm import load_llm
from app.components.vector_store import load_vector_store

from app.config.config import HUGGINGFACE_REPO_ID, HF_TOKEN
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

# ✅ Custom prompt for your medical chatbot
CUSTOM_PROMPT_TEMPLATE = """Answer the following medical question in 2-3 lines maximum using only the information provided in the context.

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt():
    """Creates a prompt template for retrieval QA."""
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

def create_qa_chain():
    """Creates a retrieval QA chain using LangChain v0.2+."""
    try:
        logger.info("Loading vector store for context")
        db = load_vector_store()

        if db is None:
            raise CustomException("Vector store not present or empty")

        llm = load_llm()

        if llm is None:
            raise CustomException("LLM not loaded")

        # ✅ Combine retriever with prompt and LLM
        prompt = set_custom_prompt()
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)

        qa_chain = create_retrieval_chain(
            retriever=db.as_retriever(search_kwargs={'k': 1}),
            combine_docs_chain=combine_docs_chain
        )

        logger.info("Successfully created the QA chain")

        # Debug (optional)
        print("\n✅ QA Chain Debug Info:")
        try:
            if hasattr(qa_chain, "input_keys"):
                print("Input keys:", qa_chain.input_keys)
            if hasattr(qa_chain, "output_keys"):
                print("Output keys:", qa_chain.output_keys)
        except Exception as e:
            print("Error printing QA chain keys:", e)
        print("✅ QA chain ready!\n")

        return qa_chain

    except Exception as e:
        error_message = CustomException("Failed to make a QA chain", e)
        logger.error(str(error_message))
        return
