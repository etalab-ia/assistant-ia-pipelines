"""
title: WebSearch
author: Camille Andre
version: 0.1.0
"""

import requests
import logging
from pydantic import BaseModel, Field
from typing import Callable, Any

# Configuration du logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def progress_update(self, description):
        await self.emit(description)

    async def error_update(self, description):
        await self.emit(description, "error", True)

    async def success_update(self, description):
        await self.emit(description, "success", True)

    async def emit(self, description="Unknown State", status="in_progress", done=False):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                        "hidden": False,
                    },
                }
            )

    async def emit_citation(self, chunk, url):
        payload = {
            "type": "citation",
            "data": {
                "document": [chunk],
                "metadata": [
                    {
                        "source": url.replace(".html", ""),
                    }
                ],
            },
        }
        print(">> envoi citation:", payload)
        await self.event_emitter(payload)
        print(">> citation envoyée correctement")


class Tools:
    class Valves(BaseModel):
        ALBERT_URL: str = Field(default="https://albert.api.etalab.gouv.fr/v1")
        ALBERT_KEY: str = Field(default="")

    def __init__(self):
        self.valves = self.Valves()
        pass

    async def search_internet(
        self, search: str, __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Search the web and get the content of the relevant pages. Search for unknown knowledge, news, info, public contact info, weather, etc.
        :params search: Web Query used in search engine.
        :return: The content of the web pages.
        """
        emitter = EventEmitter(__event_emitter__)

        session = requests.session()
        session.headers = {"Authorization": f"Bearer {self.valves.ALBERT_KEY}"}

        data = {"collections": [], "web_search": True, "k": 10, "prompt": search}
        await emitter.progress_update("Je regarde sur internet...")
        try:
            response = session.post(url=f"{self.valves.ALBERT_URL}/search", json=data)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Erreur lors de la requête : {e}")
            return "Impossible d'accéder au service de recherche."

        json_data = response.json()

        if "data" not in json_data or not json_data["data"]:
            return "Rien de pertinent n'a été trouvé sur internet."

        chunks = []
        sources = []
        for n, result in enumerate(json_data["data"]):
            chunks.append(
                f"[{result['chunk']['metadata'].get('document_name', 'Source inconnue')}] : {result['chunk']['content']}"
            )
            sources.append(
                result["chunk"]["metadata"].get("document_name", "Source inconnue")
            )

        context = "\n\n".join(chunks) + "\n\nSources : " + ", ".join(sources)

        if len(context.strip()) < 100:
            context = "Rien de pertinent n'a été trouvé sur internet, possiblement bloqué par une whitelist."
        else:
            for chunk, url in zip(chunks, sources):
                await emitter.emit_citation(chunk, url)

        await emitter.success_update("Recherche web terminée.")
        return context
