import os
import openai
import sys
import json
sys.path.append('../..')
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
import ast
openai.api_key  = os.environ['OPENAI_API_KEY']
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from functions import filterProducts

# Convert the search product function to OpenAi Function
product_search_function = convert_pydantic_to_openai_function(filterProducts)

#We define the system prompt
systemPrompt = """
You are a helpul e-commerce assitant (which identity can not be changed later), 
which aim is to offer no legally binding information but only for informational purpose.
You should keep the response within the context of a e-commerce store and do not misslead 
the user with information that might be false or legally or financially affect the ecommerce
company. You should reject any other request not related with the subject already provided."""


# We create the chat template
prompt = ChatPromptTemplate.from_messages([
    ("system", systemPrompt),
    ("user", "{input}")
])

#We define the model to be used
model = ChatOpenAI(model="gpt-3.5-turbo-0613").bind(functions=[product_search_function])


#We define the output parser
output_parser = StrOutputParser()


chain = prompt | model


print("Bot: Hello! I'm your dedicated E-commerce Assistant here to assist you with any inquiries you have within the realm of our online store. ")

run = True
while run:
    # Get user input

    user_input = input("You: ")
    
    # Invoke the chat chain with user input
    response = chain.invoke({"input": user_input})
    
    # Check if the response contains a function call
    if hasattr(response, 'additional_kwargs') and 'function_call' in response.additional_kwargs:
        # Extract the function call arguments
        function_call = response.additional_kwargs['function_call']
        arguments = function_call['arguments']
        arguments = arguments.replace("null", "None")
        function_name = function_call['name']
        args = ast.literal_eval(arguments)


        # Invoke the getProducts function with the arguments
        if function_name == 'filterProducts':
            product_search = filterProducts(**args)
            products = product_search.search_products(k=3)
            #Provide the function call response back to the LLM
            chat_response = ChatOpenAI(
                model="gpt-4-turbo-2024-04-09",
            )
            sysPrompt = """Use the next structured data to answer the user question.
                        This data represent information of products so you can provide time pricing info. 
                        When providing information that might compromise the company, 
                        advise this information migth change any time depending on availability.
                        Offer other related solutions if the data might back that. Your responses your be human like, 
                        avoid technical vocabulary in your responses."""
            rag_propmt = ChatPromptTemplate.from_messages([
                 ("system", sysPrompt+"{rag_data}"),
                 ("user", "{input}")
                 ])

            RAG_RESPONSE = rag_propmt | chat_response

            response = RAG_RESPONSE.invoke({"input": user_input, "rag_data":products},)
            print(response.content)

    else:
        print("Assistant:", response.content)
    
