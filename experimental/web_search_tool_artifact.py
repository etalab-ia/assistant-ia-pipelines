"""
title: WebSearch
author: Camille Andre
version: 0.1.2
"""

import requests
import logging
from pydantic import BaseModel, Field
from typing import Callable, Any
from urllib.parse import urlparse

# Configuration du logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

prompt_prefix = "Recherche internet terminée. Réponds à la demande ou question de l'utilisateur en te basant le contexte suivant, extrais uniquement le contenu pertinent et ignore le reste : \n"
prompt_suffix = "Ne mets pas les URLs dans ta réponse, je les donnerais moi même à l'utilisateur." #"\n A la fin de ta réponse, ajoutes en markdown uniquement trois URLs max (ou moins) présentes les plus importantes dans le contexte avec le format [Nom du site](URL). Ne fais pas de doublons dans les URLs."

def nom_site_simple(url: str) -> str:
    h = urlparse(url).hostname or url
    p = h.split('.')
    return p[-2] if len(p) >= 2 else h

def check_iframe_support(url):
    """
    Vérifie si une URL accepte d'être intégrée dans des iframes
    en analysant les en-têtes HTTP de sécurité
    """
    try:
        # Ajouter http:// si aucun schéma n'est spécifié
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # Effectuer une requête HEAD pour récupérer uniquement les en-têtes
        response = requests.head(url, timeout=10, allow_redirects=True)

        # Vérifier le code de statut
        if response.status_code >= 400:
            print(f"❌ Erreur HTTP {response.status_code} pour {url}")
            return False

        headers = response.headers

        # Analyser les en-têtes de sécurité
        results = {
            "url": url,
            "status_code": response.status_code,
            "iframe_allowed": True,
            "restrictions": [],
        }

        # Vérifier X-Frame-Options
        x_frame_options = headers.get("X-Frame-Options", "").upper()
        if x_frame_options:
            if x_frame_options in ["DENY", "SAMEORIGIN"]:
                results["iframe_allowed"] = False
                results["restrictions"].append(f"X-Frame-Options: {x_frame_options}")
            elif x_frame_options.startswith("ALLOW-FROM"):
                results["restrictions"].append(f"X-Frame-Options: {x_frame_options}")

        # Vérifier Content-Security-Policy
        csp = headers.get("Content-Security-Policy", "")
        if "frame-ancestors" in csp.lower():
            if "'none'" in csp.lower():
                results["iframe_allowed"] = False
                results["restrictions"].append("CSP: frame-ancestors 'none'")
            elif "'self'" in csp.lower():
                results["restrictions"].append("CSP: frame-ancestors 'self'")
            else:
                # Extraire la directive frame-ancestors
                csp_parts = csp.split(";")
                for part in csp_parts:
                    if "frame-ancestors" in part.lower():
                        results["restrictions"].append(f"CSP: {part.strip()}")

        return results["iframe_allowed"]

    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur de connexion pour {url}: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue pour {url}: {e}")
        return False

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
                logger.warning(
                    "Rerank failed (score threshold), returning original chunks"
                )
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

    async def emit_artifact(self, urls):
        def generate_iframes_with_client_validation(urls):
            """
            Génère les iframes avec validation côté client
            """
            html = ""
            html_ok = ""
            html_nok = ""

            for i, url in enumerate(urls):
                safe_url = url.replace(".html", "")
                if check_iframe_support(safe_url):
                    html_ok += f"""
                    <div id="container-{i}" style="width:100%;height:800px;border-radius:8px; overflow:hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.2); margin-bottom:20px;">
                      <iframe id="frame-{i}" src="{safe_url}" width="100%" height="100%" style="border:none;"></iframe>
                    </div>
                    """
                else:
                    html_nok += f"""
                        <div id="container-{i}" style="width:100%;height:60px;border-radius:8px;overflow:hidden;
                            box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:20px;display:flex;align-items:center;
                            justify-content:center;gap:12px;background:#f7f9fc;border:1px solid #e2e8f0;
                            font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#334155;">
                        <span>Aperçu indisponible</span>
                        <a href="{safe_url}" target="_blank" rel="noopener"
                            style="padding:6px 10px;border-radius:6px;text-decoration:none;font-size:.9em;
                            background:#2563eb;color:#fff;border:1px solid #1e4fd1;">
                            Ouvrir {nom_site_simple(safe_url).capitalize()}
                        </a>
                        </div>
                    """
            html = html_ok + html_nok

            return html

        html = generate_iframes_with_client_validation(urls)

        print("HTML : ", html)
        await self.event_emitter(
            {
                "type": "artifacts",
                "data": {
                    "type": "iframe",
                    "content": f"""
                {html}
                """,
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

        if "data" not in json_data:

            print(response.text)
            await emitter.success_update("Erreur lors de la recherche sur internet.")
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
            print("Erreur rerank")
            chunks_list = json_data["data"]

        print(chunks_list)
        chunks = []
        sources = []
        for n, result in enumerate(chunks_list):
            chunks.append(
                f"[{result['chunk']['metadata'].get('document_name', 'Source inconnue')}] : {result['chunk']['content']}"
            )
            sources.append(
                result["chunk"]["metadata"]
                .get("document_name", "Source inconnue")
                .removesuffix(".html")  # .html from Albert API
            )

        context = (
            prompt_prefix
            + "\n\n".join(chunks)
        #    + "\n\nSources : "
        #    + ", ".join(sources)
            + prompt_suffix
        )

        if len(chunks) == 0:
            context = "Rien de pertinent n'a été trouvé sur internet, Dis à l'utilisateur que le tool internet est bloqué par une whitelist ce qui empêche l'accès à certains sites web."
        else:
            for chunk, url in zip(chunks, sources):
                await emitter.emit_citation(chunk, url)

        await emitter.success_update("Recherche internet terminée.")
        sources = list(set(sources))
        await emitter.emit_artifact(sources)

        return context
