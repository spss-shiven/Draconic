import os
import yaml
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

os.environ["GEMINI_API_KEY"] = ""

provider = GoogleGLAProvider(api_key=os.environ["GEMINI_API_KEY"])
model = GeminiModel("models/gemini-pro", provider=provider)


def load_agents_from_yaml(file_path):
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    agents = {}
    for agent_cfg in config["agents"]:
        model = GeminiModel("gemini-2.5-pro")
        agent = Agent(
            model=model,
            system_prompt=agent_cfg["system_prompt"],
            tools=[]
        )
        agents[agent_cfg["name"]] = agent
    return agents


query = """{
            "article_id": "FIN-001",
            "headline": "Tesla crushes Q3 expectations with record profits, but Musk warns of 'turbulent t
            "content": "Tesla (NASDAQ: TSLA) reported stunning Q3 results with earnings of $1.05 per share
            "published_at": "2024-10-22T16:00:00Z"
}"""
agents = load_agents_from_yaml("/content/sample_data/agents.yaml")
prompt_for_analyzer = "The query is as follows: " + query 
analysis = agents["analyzer"].run_sync(prompt_for_analyzer)

prompt_for_assesser = "Assess the following analysis, analysis is as follows: " + analysis.data
final_output = agents["assesser"].run_sync(prompt_for_assesser)
print(final_output.data)
