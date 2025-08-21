"""
title: WebSearch
author: Camille Andre
version: 0.1.2
"""

import requests
import logging
from pydantic import BaseModel, Field
from typing import Callable, Any, Optional


# Configuration du logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

prompt_prefix = "Réponds à la demande ou question de l'utilisateur en te basant le contexte des sources suivantes :"
prompt_suffix = "\n A la fin de ta réponse, ajoutes en markdown uniquement trois URLs max présentes les plus importantes dans le contexte avec le format [Nom du site](URL). Ne fais pas de doublons dans les URLs."


def reranker(
    query: str,
    chunks: list,
    api_url: str,
    api_key: str = "",
    score_threshold: Optional[float] = 0,
    rerank_model: str = "BAAI/bge-reranker-v2-m3",
    min_chunks: int = 1,
):
    """Reorder documents by relevance using Albert's reranking API."""
    chunks_questions = []

    chunks_questions.extend((chunk.get("chunk").get("content")) for chunk in chunks)

    request_body = {
        "prompt": query,
        "input": chunks_questions,
        "model": rerank_model,
    }

    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.post(
        url=f"{api_url}/rerank", json=request_body, headers=headers
    )

    if response.status_code == 200:
        results = response.json()
        if "data" in results:
            # Get indices sorted by highest score
            ranked_indices = [item["index"] for item in results["data"]]
            
            if score_threshold is not None:
                reranked_chunks = [
                    chunks[idx]
                    for idx in ranked_indices
                    if results["data"][idx]["score"] >= score_threshold
                ]
            else:
                reranked_chunks = [chunks[idx] for idx in ranked_indices]
            
            # Fallback to original chunks if too few results after filtering
            if len(reranked_chunks) < min_chunks:
                logger.warning("Rerank failed (score threshold), returning original chunks")
                return chunks
            else:
                logger.info("Rerank success, returning reranked chunks")
                return reranked_chunks
        else:
            logger.error(f"Format de réponse inattendu: {results}")
            return chunks
    else:
        logger.error(f"Erreur lors du reranking: {response.status_code}")
        return chunks


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
                        "source": url,
                    }
                ],
            },
        }
        await self.event_emitter(payload)


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
        Recherche sur Internet pour récupérer le contenu de sites. Cherche des informations, des nouvelles, des connaissances, des contacts publics, la météo, etc.
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
            await emitter.success_update("Erreur du service de recherche.")
            return "Impossible d'accéder au service de recherche, limite d'utilisation journalière dépassée."

        json_data = response.json()

        if "data" not in json_data or not json_data["data"]:
            await emitter.success_update(
                "Erreur lors de la recherche sur internet."
            )
            return "Erreur lors de la recherche sur internet."

        # Rerank results for better relevance
        try:
            chunks_list = reranker(
                        query=search,
                        chunks=json_data["data"],
                        api_url=self.valves.ALBERT_URL,
                        api_key=self.valves.ALBERT_KEY,
                        min_chunks=5,
                    )[:5]
        except Exception as e:
            logger.error(f"Erreur lors du reranking: {e}")
            chunks_list = json_data["data"]

        chunks = []
        sources = []
        for n, result in enumerate(chunks_list):
            chunks.append(
                f"[{result['chunk']['metadata'].get('document_name', 'Source inconnue')}] : {result['chunk']['content']}"
            )
            sources.append(
                result["chunk"]["metadata"].get("document_name", "Source inconnue").removesuffix('.html')  # .html from Albert API
            )

        context = (
            prompt_prefix
            + "\n\n".join(chunks)
            + "\n\nSources : "
            + ", ".join(sources)
            + prompt_suffix
        )

        if len(chunks) == 0 :
            context = "Rien de pertinent n'a été trouvé sur internet, possiblement bloqué par une whitelist."
        else:
            for chunk, url in zip(chunks, sources):
                await emitter.emit_citation(chunk, url)

        await emitter.success_update("Recherche internet terminée.")

        return context
