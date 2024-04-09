from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.chroma import Document
import csv
import os
import json

class filterProducts(BaseModel):
    """
    Call this to get a list of products that match the given product properties.
    You can provide one or several product proterties. 
    returns:
    [{
        product: string,
        type_id: number,
        size: string,
        color: string,
        price: string,
        inventory: { stock: number; location: string }[],
        sku: string;
    }]
    """
    name: Optional[str] = Field(description="A product name to apply the filter")
    type: Optional[str] = Field(description="A product type such as shirt, pant, shoe...")
    size: Optional[str] = Field(description="A product size S, M, L...")
    location: Optional[str] = Field(description="A location for the product size S, M, L...")
    price: Optional[str] = Field(description="A price for the product.")
    color: Optional[str] = Field(description="A color for the product.")

    embeddings: Optional[str] = Field(description="Not required")
    vectorstore: Optional[str] = Field(description="Not required")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = self.create_vectorstore()

    def create_vectorstore(self):
        """
        Create a Chroma vector store from the 'data.csv' file.
        """
        metadatas = []
        with open('data.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                metadata = {
                    "product": row['product'],
                    "type_id": row['type_id'],
                    "size": row['size'],
                    "color": row['color'],
                    "price": row['price'],
                    "inventory": row['inventory'],
                    "description": row['description'],
                    "sku": row['sku']
                }
                metadatas.append(metadata)

        vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=os.path.join(os.getcwd(), "vectorstore")
        )
        
        # Add each metadata dictionary as a document object to the vector store
        # Add each metadata dictionary as a document object to the vector store
        for metadata in metadatas:
            # Convert metadata to JSON string and add as text document
            metadata_str = json.dumps(metadata)
            vectorstore.add_texts([metadata_str])
        
        return vectorstore


    def search_products(self, k=8):
        """
        Search for products that match the given properties using RAG.
        """
        query = ""
        if self.name:
            query += f"product:{self.name} "
        if self.type:
            query += f"type_id:{self.type} "
        if self.size:
            query += f"size:{self.size} "
        if self.location:
            query += f"inventory:{self.location} "
        if self.price:
            query += f"price:{self.price} "

        docs = self.vectorstore.similarity_search(query, k=k)
        products = []
        for doc in docs:
            product_data = eval(doc.page_content)
            products.append(product_data)
        return products
    
