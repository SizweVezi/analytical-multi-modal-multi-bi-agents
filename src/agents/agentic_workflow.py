from src.main import SystemMessage, HumanMessage, ToolMessage, AnyMessage
from src.utils.api_helpers import llm

#####
# Router agent
#####
question_category_prompt = '''You are a senior specialist of Financial Analytical Support. Your task is to classify the incoming questions. 
Depending on your answer, question will be routed to the right team, so your task is crucial for our team. 
There are 5 possible question types: 
- **audio_earning_call** - Answer questions related to pre-indexed Amazon earning call related topics stored in the vactor store.
- **Image_Search**- Answer questions related with image, figure, or diagram search about company's financial performance
- **company_financial**- Answer questions based on Company Financial information, such as stock information, income statement, stock volatility, etc.
- **chat**- Answer questions for LLM or a few LLMs.
- **financial_report**- Answer questions, writing a financial report. use retrived information to create the report.
If the intent isn't clear or doesn't match any specific category, use **chat**.
Return in the output only one word (audioearningcall, Image_Search, CompanyFinancial, chat, financial_report).

'''

def router_node(state: MultiAgentState):
    print('Router node started execution')
    messages = [
        SystemMessage(content=question_category_prompt),
        HumanMessage(content=state['question'])
    ]
    response = llm.invoke(messages)
    print('Question type: %s' % response.content)
    return {"question_type": response.content}


#######
# AUDIO RAG
#######
def rag_node(state: MultiAgentState):
    """
    RAG node function using Amazon Knowledge Bases with audio segment playback
    Args:
        state: MultiAgentState containing the question
    Returns:
        dict: Contains the answer, response object, and audio processing results
    """
    try:
        # Initialize the Knowledge Base retriever
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=AUDIO_KB_ID,
            retrieval_config={
                "vectorSearchConfiguration": {
                    "numberOfResults": 3
                }
            }
        )

        # Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        # Get the question from state
        question = state['question']

        # Get response from the chain
        response = qa_chain(question)

        # Extract the generated answer
        generation = response['result']

        # Extract audio information and timestamps from the response
        audio_s3_info, timestamps = extract_audio_path_and_timestamps(response)

        # Store audio information in state for later use
        state['audio_info'] = {
            's3_info': audio_s3_info,
            'timestamps': timestamps
        }
        state['audio_processed'] = True
        state['has_audio'] = bool(audio_s3_info and timestamps)
        state['raw_response'] = response

        # Return enhanced response with all information
        return {
            'answer': generation,
            'raw_response': response,  # Include full response for downstream processing
        }

    except Exception as e:
        print(f"Error in RAG node: {e}")
        return {
            'answer': f"Error occurred while processing the question: {str(e)}",
            'raw_response': None,
            'audio_info': None
        }


#######
# IMAGE RAG
#######
def rag_node_image(state: MultiAgentState):
    """
    RAG node function using Amazon Knowledge Bases for image retrieval
    Args:
        state: MultiAgentState containing the question
    Returns:
        dict: Contains the answer, response object, and image citations
    """
    try:
        # Initialize the Knowledge Base retriever
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=IMAGE_KB_ID,
            retrieval_config={
                "vectorSearchConfiguration": {
                    "numberOfResults": 3
                }
            }
        )

        # Get the question from state
        question = state['question']

        try:
            # Get documents using invoke
            retrieved_docs = retriever.invoke(question)

            if not retrieved_docs:
                raise ValueError("No documents retrieved")

            # Process retrieved documents
            if isinstance(retrieved_docs, list):
                # Extract content and citations from retrieved documents
                citations = []
                content = []

                for doc in retrieved_docs:
                    if hasattr(doc, 'metadata') and 'citations' in doc.metadata:
                        citations.extend(doc.metadata['citations'])
                    if hasattr(doc, 'page_content'):
                        content.append(doc.page_content)

                response = {
                    'result': '\n'.join(content) if content else '',
                    'citations': citations,
                    'source_documents': retrieved_docs
                }
            else:
                response = retrieved_docs

        except Exception as chain_error:
            print(f"Retrieval error: {chain_error}")
            return {
                'answer': f"Error retrieving documents: {str(chain_error)}",
                'raw_response': None
            }

        # Store response in state
        state['raw_response'] = response

        # Extract and store image citations
        citations = response.get('citations', [])
        if citations:
            state['has_images'] = True
            state['image_citations'] = citations
        else:
            state['has_images'] = False
            state['image_citations'] = None

        # Update state instead of returning new dict
        state['answer'] = response.get('result', '')
        state['raw_response'] = response

        return state  # Return the entire state object

    except Exception as e:
        print(f"Error in RAG node: {e}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {str(e)}")

        # Update state with error information
        state['answer'] = f"Error occurred while processing the question: {str(e)}"
        state['raw_response'] = None
        return state  # Return the entire state object