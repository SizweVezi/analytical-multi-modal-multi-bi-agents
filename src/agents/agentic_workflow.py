from src.main import SystemMessage, HumanMessage, ToolMessage, AnyMessage
from src.utils.api_helpers import llm
from PIL import Image
import os
import base64
import io
import matplotlib.pyplot as plt

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


#####
# LLM node
####
def llm_node(state: MultiAgentState):
    model_ids = [model_id_mistral_large, model_id_c35]
    max_tokens = 2048
    temperature = 0.01
    top_p = 0.95

    conversation = [
        {
            "role": "user",
            # "system": "You are a domain expert who can understand the intent of user query and answer question truthful and professionally. Please, don't provide any unchecked information and just tell that you don't know if you don't have enough info.",
            "content": [{"text": state['question']}],
        }
    ]
    try:
        # Send the message to the model, using a basic inference configuration.
        responses = []
        for model_id in model_ids:
            response = bedrock_client.converse(
                modelId=model_id,
                messages=conversation,
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature, "topP": top_p},
            )

            # Extract and print the response text.
            responses.append(response["output"]["message"]["content"][0]["text"])

        ###
        # Combine the answers to form a unified one
        ###
        c3_template = """Your are a domain expert and your goal is to Merge and eliminate redundant elements from {{responses}} that captures the essence of all input while adhering to the following the {{instruction}}.
        <instructions> 
            <step>Aggregate relevant information from the provided context.</step> 
            <step>Eliminate redundancies to ensure a concise response.</step> 
            <step>Maintain fidelity to the original content.</step> 
            <step>Add additional relevent info to the question or removing iirelevant information.</step>
        </instructions> 
        <responses>
            {responses}
        </responses>
        """

        messages = [
            SystemMessage(content=c3_template),
            HumanMessage(content=state['question'])
        ]

        return {'answer': llm.invoke(messages)}
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")


#####
# Image generation node
####
def t2i_node(state: MultiAgentState):
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    model_id = "amazon.nova-canvas-v1:0"

    prompt = f"Generate a high resolution, photo realistic picture of {state['question']} with vivid color and attending to details."

    accept = "application/json"
    content_type = "application/json"

    # Format the request payload
    request_payload = json.dumps({
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,

        }
    })

    try:

        # Invoke the model
        response = client.invoke_model(
            body=request_payload, modelId=model_id, accept=accept, contentType=content_type
        )

        # Parse the response
        response_body = json.loads(response.get("body").read())

        # Extract the base64 image data
        base64_image = response_body.get("images")[0]
        base64_bytes = base64_image.encode('ascii')
        image_bytes = base64.b64decode(base64_bytes)

        finish_reason = response_body.get("error")

        with open(temp_gen_image, 'wb') as file:
            file.write(image_bytes)

        return {"answer": temp_gen_image}

    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return {"answer": f"Error generating image:  {str(e)}"}


#####
# BlogWriter node
####

def financial_report_node(state: MultiAgentState) -> MultiAgentState:
    """financial_report node for the workflow"""
    try:
        # Get the question from state
        question = state.get('question', '')
        if not question and 'answer' in state:
            question = state['answer']

        # Initialize bedrock client
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1"
        )

        # Initialize the model
        llm = ChatBedrock(
            model_id=model_id,
            client=bedrock_client,
            model_kwargs={
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_p": 0.95,
            }
        )

        # Create the financial_report writing prompt
        blog_prompt = f"""Write a comprehensive and engaging financial report about: {question}

        Please ensure the financial report:
        1. Has a clear structure with introduction, body, and conclusion
        2. Includes relevant facts and figures where appropriate
        3. Is written in an engaging, professional style
        4. Incorporates current trends and developments
        5. Is optimized for readability
        """

        # Get response from LLM
        response = llm.invoke(blog_prompt)

        # Initialize messages if not present
        if 'messages' not in state:
            state['messages'] = []

        # Update state with response
        content = response.content if hasattr(response, 'content') else str(response)
        state['messages'].append(HumanMessage(content=content))
        state['answer'] = content

        return state

    except Exception as e:
        print(f"Error in financial_report_node: {str(e)}")
        if 'messages' not in state:
            state['messages'] = []
        state['messages'].append(HumanMessage(content=f"Error: {str(e)}"))
        state['answer'] = f"Error: {str(e)}"
        return state


#####
# Hallucination grader
#####

def should_end(state):
    """Determine if we should end the conversation"""
    # Check for explicit END signal
    if state.get('next') == 'END':
        return True

    # Check for __end__ flag in results
    if isinstance(state.get('results'), dict) and state['results'].get('__end__', False):
        return True

    return False


def handle_error(state: MultiAgentState) -> bool:
    """
    Checks if there's an error in the state
    Returns True if there's an error, False otherwise
    """
    return state.get("error") is not None


class MyCustomHandler(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        print(f"Response: {response}")


def hallucination_grader(state: MultiAgentState):
    c3_template = """You are a grader assessing whether an answer is grounded in supported by facts. 
        Give a binary score 'pass' or 'fail' score to indicate whether the answer is grounded in supported by a 
        set of facts in your best knowledge. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

        Here is the answer: {answer}"""
    c3_prompt = ChatPromptTemplate.from_template(c3_template)

    # Grade by a diff model in this case Claude 3
    # hallucination_grader = prompt | llm_llama31  | JsonOutputParser()
    hallucination_grader = c3_prompt | llm_claude35 | JsonOutputParser()
    score = hallucination_grader.invoke({"answer": state['answer'], "callbacks": [MyCustomHandler()]})
    if "yes" in score['score'].lower():
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: the answer does not seem to contain hallucination ---"
        )
        return "END"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: the answer migh contain hallucination, next off to human review ---")
        return "to_human"


####
# Extra function but not as a node
####
def decide_to_search(state: MultiAgentState):
    """
    Determines whether to generate an answer, or add web search
    Args:
        state (dict): The current graph state
    Returns:
        str: Binary decision for next node to call
    """
    l31_prompt = PromptTemplate(
        template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
        an {answer} is grounded in / relevant to the {question}. Give a binary score 'yes' or 'no' score to indicate
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
        single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the answer:
        {answer}
        Here is the question: {question}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "answer"],
    )

    answer_grader = l31_prompt | llm_mistral | JsonOutputParser()
    print("---ASSESS GRADED ANSWER AGAINST QUESTION---")
    relevance = answer_grader.invoke({"answer": state["answer"], "question": state["question"]})
    print(relevance)
    if "yes" in relevance['score'].lower():
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: the answer is relevant to the question so it's ready for human review ---"
        )
        return "to_human"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: the answer is NOT relevant to the question then try LLM route---")
        return "do_search"


def where_to(state: MultiAgentState) -> str:
    """
    Routes to different branches based on question_type from router
    """
    print('where_to')
    print('State', state)

    question_type = state.get('question_type', '')

    if question_type == 'Image_Search':
        return 'Image_Search'
    elif question_type in ['CompanyFinancial', 'company_financial']:  # Fixed the condition
        return 'company_financial'
    elif question_type == 'audioearningcall':
        return 'audioearningcall'
    elif question_type == 'financial_report':
        return 'financial_report'
    else:
        return 'chat'


# Create the workflow
workflow = StateGraph(MultiAgentState)

# Add nodes
#workflow.add_node("rewriter", rewrite_node)
workflow.add_node("router", router_node)
workflow.add_node("ChatNode", llm_node)
workflow.add_node("company_financial", reasoner)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("audioearningcall_expert", rag_node)
workflow.add_node("Image_Search", rag_node_image)
workflow.add_node('financial_report', financial_report_node)
workflow.add_node('text2image_generation', t2i_node)

# Basic flow
workflow.add_edge(START, "router")
#workflow.add_edge("rewriter", "router")

# Router conditional edges
workflow.add_conditional_edges(
    "router",
    where_to,
    {
        'company_financial': 'company_financial',
        'chat': 'ChatNode',
        'audioearningcall': 'audioearningcall_expert',
        'Image_Search': 'Image_Search',
        'Text2Image': 'text2image_generation',
        'financial_report':'financial_report',
    }
)

# Company financial and tools interaction
workflow.add_conditional_edges(
    "company_financial",
    tools_condition,
    {
        "tools": "tools",    # If tool needed, go to tools
        "END": END          # If no tool needed, end
    }
)


workflow.add_edge("tools", "company_financial")

# audio earning call expert routing
workflow.add_conditional_edges(
    "audioearningcall_expert",
    decide_to_search,
    {
        "to_human": END,
        "do_search": "ChatNode"
    }
)

# Add normal end connections for other paths
workflow.add_edge('ChatNode', END)
workflow.add_edge('Image_Search', END)
workflow.add_edge('financial_report', 'text2image_generation')
workflow.add_edge('text2image_generation', END)

# Compile the workflow
app = workflow.compile()


thread = {"configurable": {"thread_id": "42",  "recursion_limit": 10}}
results = []
prompts =[
        "Give me a summary of Amazon's Q3 2024 earning based on the earning call audio", # Use native RAG then human review if needed
        ]

for prompt in prompts:
    for event in graph.stream({'question':prompt,}, thread):
        print(event)
        results.append(event)
    print("\n\n---------------------------------------\n\n")

# Extract audio path and timestamps from the response
audio_s3_info, timestamps = extract_audio_path_and_timestamps_agent_response(results)
print("\nExtracted S3 info:", audio_s3_info)
print("Number of timestamps extracted:", len(timestamps))
if timestamps:
    print("\nFirst timestamp entry:", timestamps[0])


# Extract video path and timestamps from the response
play_audio_segments_from_s3(audio_s3_info, timestamps)

saved_result = None
from PIL import Image


def show_result(prompt):
    global saved_result
    try:
        thread = {
            "configurable": {
                "thread_id": "42",
                "recursion_limit": 10,
                "checkpoint_ns": "default",
                "checkpoint_id": "1"
            }
        }

        saved_result = graph.invoke({'question': prompt}, config=thread)

        # Immediately process and display the image
        if saved_result and 'answer' in saved_result:
            answer = str(saved_result['answer'])

            if answer.startswith('data:image/png;base64,'):
                # Extract and clean base64 string
                base64_str = answer.replace('data:image/png;base64,', '').strip()

                # Add padding if needed
                padding = len(base64_str) % 4
                if padding:
                    base64_str += '=' * (4 - padding)

                try:
                    # Decode and display image
                    image_bytes = base64.b64decode(base64_str)
                    image = Image.open(io.BytesIO(image_bytes))

                    plt.figure(figsize=(10, 10))
                    plt.imshow(image)
                    plt.axis('off')
                    plt.show()

                except Exception as e:
                    print(f"Image processing error: {e}")

    except Exception as e:
        print(f"Error: {e}")


# Run the function to immediately show the image
prompt = "Show me diagrams of Amazon TTM operation income and net sales in 2024"
show_result(prompt)


thread = {"configurable": {"thread_id": "42",  "recursion_limit": 10}}
results = []
prompts =[
        "create a financial report based on Amazon latest results ", # Use native RAG then human review if needed
        ]

for prompt in prompts:
    for event in graph.stream({'question':prompt,}, thread):
        print(event)
        results.append(event)
        if os.path.exists(temp_gen_image):
            Image.open(temp_gen_image).show()
    print("\n\n---------------------------------------\n\n")

thread = {
    "configurable": {
        "thread_id": "42",
        "recursion_limit": 10
    }
}

results = []
prompts = [
    "How about Uber's stock performance?"
]

for prompt in prompts:
    try:
        for event in graph.stream({'question': prompt}, thread):
            print(event)
            results.append(event)

            # Check for end condition in results

            if isinstance(event, dict) and event.get('__end__', False):
                break
    except KeyError as e:
        if str(e) == "'__end__'":
            # Expected end of stream, continue to next prompt
            pass
        else:
            print(f"Unexpected KeyError: {e}")
    except Exception as e:
        print(f"Error: {e}")
    print("\n\n---------------------------------------\n\n")