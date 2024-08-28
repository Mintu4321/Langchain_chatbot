from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
import docker
import pandas as pd
import csv

# Load environment variables from .env file
load_dotenv()

# Configure the generative AI client
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize the language model
llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.7)

# Initialize Docker client
client = docker.from_env()

def docker_monitor():
    """
    Retrieve a list of all Docker containers (running and stopped).
    """
    containers = client.containers.list(all=True)  # List all containers
    return containers

def get_container_logs(container, max_lines=10):
    """
    Retrieve the logs of a Docker container, limiting to a specified number of lines.
    """
    try:
        logs = container.logs(tail=max_lines).decode('utf-8')  # Retrieve last 'max_lines' lines of logs
        return logs
    except Exception as e:
        return f"Error retrieving logs: {e}"

def docker_inspect():
    """
    Inspect each Docker container and return its attributes and logs.
    """
    container_list = docker_monitor()
    container_attributes = []

    for container in container_list:
        inspect = container.attrs
        container_logs = get_container_logs(container)
        container_attributes.append({
            'ID': container.short_id,
            'Name': container.name,
            'Status': inspect['State']['Status'],
            'Image': inspect['Config']['Image'],
            'Created At': inspect['Created'],
            'Started At': inspect['State'].get('StartedAt', 'N/A'),
            'Finished At': inspect['State'].get('FinishedAt', 'N/A'),
            'Logs': container_logs
        })

    return container_attributes

def create_dataframe(container_attributes):
    """
    Create a Pandas DataFrame from the Docker container attributes.
    """
    df = pd.DataFrame(container_attributes)
    return df

def save_to_csv(df, filename="docker_analysis_output.csv"):
    """
    Save the DataFrame to a CSV file.
    """
    df.to_csv(filename, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"DataFrame saved to {filename}")

def llm_monitor():
    """
    Use LLM to analyze Docker containers and provide a summary.
    """
    container_attributes = docker_inspect()
    df = create_dataframe(container_attributes)

    # Convert DataFrame to a string for LLM input
    container_summaries_str = df.to_string(index=False)

    # Define the prompt template with a concise instruction
    prompt_template = PromptTemplate(
        input_variables=['docker_containers'],
        template=(
            "Analyze the following Docker containers, providing a concise summary "
            "of their status, potential issues, recommendations and docker logs. Limit details to avoid token overflow. "
            "Here are the containers and their last few logs: {docker_containers}"
        )
    )

    # Create the LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Generate the output by passing the string input
    output = llm_chain.run({'docker_containers': container_summaries_str})

    # Print or return the output for debugging
    print(output)

    # Convert the LLM output to a DataFrame
    output_df = pd.DataFrame({"LLM Analysis": [output]})

    # Save both DataFrames to a CSV
    final_df = pd.concat([df, output_df], axis=1)
    save_to_csv(final_df)

    return output

if __name__ == "__main__":
    llm_monitor()
