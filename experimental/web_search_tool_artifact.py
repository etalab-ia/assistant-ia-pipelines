"""
title: WebSearch
author: Camille Andre
version: 0.1.0
"""

import requests
import logging
from pydantic import BaseModel, Field
from typing import Callable, Any


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
                    <div id="container-{i}" style="width:100%;height:60px;border-radius:8px; overflow:hidden; 
                         box-shadow: 0 4px 12px rgba(0,0,0,0.2); margin-bottom:20px; 
                         display:flex; align-items:center; justify-content:center; background-color:#fee; border:1px solid #f00; font-family:sans-serif; color:#900;">
                        ⚠ Aperçu indisponible
                        <a href="{safe_url}" target="_blank" style="margin-left:10px; padding:4px 8px; background:#027ad6; color:#fff; border-radius:4px; text-decoration:none; font-size:0.9em;">
                            Ouvrir dans un nouvel onglet
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
        Search the web and get the content of the relevant pages. Search for unknown knowledge, news, info, public contact info, weather, etc.
        :params search: Web Query used in search engine.
        :return: The content of the web pages.
        """
        emitter = EventEmitter(__event_emitter__)

        session = requests.session()
        session.headers = {"Authorization": f"Bearer {self.valves.ALBERT_KEY}"}

        data = {"collections": [], "web_search": True, "k": 10, "prompt": search}
        await emitter.progress_update(f"Je regarde sur le net...")
        try:
            response = session.post(url=f"{self.valves.ALBERT_URL}/search", json=data)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Erreur lors de la requête : {e}")
            await emitter.success_update("Erreur du service de recherche.")
            return "Impossible d'accéder au service de recherche."

        json_data = response.json()

        if "data" not in json_data or not json_data["data"]:
            await emitter.success_update(
                "Rien de pertinent n'a été trouvé sur internet."
            )
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

        url = sources[0]
        # print("URL : ", urls)
        sources = list(set(sources))
        await emitter.emit_artifact(sources)

        return context
