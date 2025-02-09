import streamlit as st
import uuid
from typing import List, Dict, Any
from dataclasses import dataclass
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="llama2", temperature=0)

@dataclass
class SearchResult:
    doc_id: str
    content: str
    score: float

class RAGChatbot:
    def __init__(self, model_name: str = "mxbai-embed-large", persist_dir: str = "ollama"):
        self.embeddings = OllamaEmbeddings(
            model=model_name,
        )
        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_dir
        )
        
        self.query_generation_template = """
         I use you to generate different variations of a given question to improve search results.
         Provide me with just a list of the questions.
         Generate 3 different versions of the following question that capture different aspects of the query.
         Format response as a Python list like [question1, question2, question3]
         Strictly follow the format, otherwise, the response will be incorrect.
         whole your answer will be evaluated as python code.
         Original question: {question}
         """

        self.response_synthesis_template = """
        Synthesize these responses into a single coherent answer:
        Responses: {responses}
        
        Ensure the answer is:
        1. Comprehensive yet concise
        2. Well-structured
        3. Directly addresses the original question
        
        Answer:
        """
        
        self.rag_template = """
        Context: {context}
        
        Question: {question}
        
        Using the above context, provide a clear and accurate answer to the question.
        If the context doesn't contain enough information, say so.
       " 
        Answer:
        """

    def generate_query_variations(self, question: str) -> List[str]:
        prompt = PromptTemplate(
            input_variables=["question"],
            template=self.query_generation_template
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(question=question)
        print(response)
        parts = response.split(':')

        
        if len(parts) > 1:
            # question = parts[0].strip()  # Part before the colon
            answer = parts[1].strip()
            answer = answer.split('=')
            answer = answer[1].strip()
            answer = answer.split(']')
            answer = answer[0].strip()
            answer = answer + ']'

        print(answer)
        variants = eval(answer)

        return list(set(variants))
        # try:
        #     print("************************************************")
        #     print(response)
        #     # Ensure the response is a valid Python list
        #     if response.startswith('[') and response.endswith(']'):
        #         variants = eval(response)
        #     else:
        #         raise ValueError("Response is not a valid Python list")
        #
        #     if not isinstance(variants, list):
        #         variants = [question]
        #     variants.append(question)
        #     return list(set(variants))
        # except Exception as e:
        #     st.warning(f"Error generating query variations: {e}")
        #     return [question]

    def reciprocal_rank_fusion(
        self,
        results: List[SearchResult],
        k: float = 60.0
    ) -> List[SearchResult]:
        doc_scores: Dict[str, float] = {}
        
        for rank, result in enumerate(results):
            score = 1 / (k + (rank + 1))
            if result.doc_id not in doc_scores:
                doc_scores[result.doc_id] = score
            else:
                doc_scores[result.doc_id] += score
        
        sorted_results = sorted(results, 
                              key=lambda x: doc_scores[x.doc_id],
                              reverse=True)
        
        return [
            SearchResult(
                doc_id=r.doc_id,
                content=r.content,
                score=doc_scores[r.doc_id]
            )
            for r in sorted_results
        ]

    def search_documents(
        self,
        query: str,
        n_results: int = 3
    ) -> List[SearchResult]:
        query_variations = self.generate_query_variations(query)
        all_results: List[SearchResult] = []
        
        for variant in query_variations:
            results = self.vector_store.similarity_search_with_score(
                variant,
                k=n_results
            )
            
            for doc, score in results:
                all_results.append(
                    SearchResult(
                        doc_id=str(uuid.uuid4()),
                        content=doc.page_content,
                        score=score
                    )
                )
        
        fused_results = self.reciprocal_rank_fusion(all_results)
        
        seen = set()
        unique_results = []
        for result in fused_results:
            if result.content not in seen:
                seen.add(result.content)
                unique_results.append(result)
                if len(unique_results) >= n_results:
                    break
        
        return unique_results

    def generate_response(self, question: str, context: List[SearchResult] = None) -> str:
        if context:
            context_text = "\n".join(r.content for r in context)
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=self.rag_template
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            return chain.run(context=context_text, question=question)
        
        query_variations = self.generate_query_variations(question)
        responses = []
        
        for query in query_variations:
            chain = LLMChain(llm=llm,
                prompt=PromptTemplate(
                    input_variables=["question"],
                    template="{question}"
                )
            )
            responses.append(chain.run(question=query))
        
        if responses:
            prompt = PromptTemplate(
                input_variables=["responses"],
                template=self.response_synthesis_template
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            return chain.run(responses=responses)
        
        return "I apologize, but I was unable to generate a response."

    def add_document(self, content: str) -> None:
        try:
            self.vector_store.add_texts(
                [content],
                ids=[str(uuid.uuid4())]
            )
        except Exception as e:
            raise Exception(f"Error adding document: {str(e)}")

def initialize_streamlit():
    st.set_page_config(
        page_title="Enhanced RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = RAGChatbot()

def main():
    initialize_streamlit()
    st.title("🤖 Enhanced RAG Chatbot")
    
    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["txt"],
            help="Upload text documents to enhance the chatbot's knowledge."
        )
        
        if uploaded_file:
            try:
                content = uploaded_file.read().decode()
                st.session_state.chatbot.add_document(content)
                st.success("Document successfully indexed!")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
    
    st.container()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        search_results = st.session_state.chatbot.search_documents(prompt)
        
        with st.chat_message("assistant"):
            response = st.session_state.chatbot.generate_response(
                prompt,
                context=search_results if search_results else None
            )
            st.write(response)
            
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        conversation = f"User: {prompt}\nAssistant: {response}"
        st.session_state.chatbot.add_document(conversation)

if __name__ == "__main__":
    main()
