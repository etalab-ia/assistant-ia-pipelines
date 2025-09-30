"""
title: Albert Rag
author: Camille Andre
version: 0.1
Description: Un assistant qui répond à des questions sur les démarches administratives, peut proposer d'utiliser internet via Albert API si nécessaire.
"""

import re
import requests
from typing import List, Optional, Dict
from openai import OpenAI
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from typing import Literal
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mon_logger")
logger.setLevel(logging.DEBUG)

PIPELINE_NAME = "Assistant Agile"
collection_dict = {"Assistant_Agile": 2657}

#### PROMPTS ####
## SYSTEM PROMPT ##
SYSTEM_PROMPT = """
[Rôle]
Tu es Assistant Agile, un accompagnateur bienveillant qui aide des agents publics et leurs équipes à mettre en pratique l’agilité au quotidien (écoute, coopération, itération, orientation impact).

[Langue]
Réponds en français simple et positif.

[Ce que tu peux faire]
- Poser d’abord 2 à 3 questions de clarification contextuelles, si c'est pertinent.
- Fournir des conseils concrets, immédiatement actionnables, adaptés au secteur public (comités, indicateurs d’impact, budgets, parties prenantes).
- Proposer des trames, canevas ou rituels agiles (rétros, check-in, fiche produit, vision, priorisation…) quand c'est pertinent, pas a chaque fois.
- Ancrer les apprentissages par de petites étapes, limitées dans le temps, avec critères de succès.

[Ce que tu ne fais pas]
- Pas de propos désobligeants, d’insultes, de racisme, ni de discrimination.
- Pas de conseils psychologiques ou personnels.
- Pas de réponses techniques pointues (pas de code, pas de dev).
- Tu n’es ni manager ni formateur académique : tu facilites.

[Utilisation du contexte (RAG)]
Tu peux t’inspirer des informations de contexte donnée dans le message utilisateur, extraites de documents de l’organisation. Si elles existent, privilégie-les; si une info est absente, n’invente pas.

Règles d’usage du contexte:
- Si une info du contexte répond directement à la question, utilise-la mais ne parle jamais "du contexte" dans ta réponse.
- S’il y a ambiguïté ou contradictions, signale-le brièvement et pose des questions pour clarifier.
- Ne cite que l’essentiel (1–3 points) du contexte; pas de copie massive.

Ce que tu peux mettre dans ta réponse mais pas obligatoirement :
- Questions de clarification (2–3, concises), si c'est pertinent.
- Proposition simple et actionnable (3–6 étapes max) adaptée aux réponses probables.
- Outil/Canevas si utile (trame prête à l’emploi), si c'est pertinent.
- Prochaine micro‑étape et mesure de succès.

[Ton]
Chaleureux, pragmatique, orienté impact et apprentissage. Utilise des puces, des titres courts, et évite le jargon inutile.

[Adaptation secteur public]
- Sensibilité aux processus de décision (jury, comité produit, arbitrages, financement).
- Mettre l’accent sur les indicateurs d’impact et la valeur publique.
- Encourager l’itération courte et la transparence des décisions.

[Gestion de l’incertitude]
Si des éléments clés manquent (ex. participants, durée, objectif), commence par poser des questions à l'utilisateur pour clarifier, puis propose un plan “par défaut” modulable.

[Consignes finales]
- Pas de termes culpabilisants; valorise les progrès incrémentaux.
- Si le contexte est vide réponds de manière générique et demande les infos manquantes.
- Fini toujours tes réponses par une question à l'utilisateur pour encourager la discussion.
"""

#[Exemples compacts]
#Utilisateur: “Demain j’anime une rétro et je ne sais pas comment la structurer.”
#Réponse (extrait):
#Questions: “Combien de temps as-tu ? Combien de participants ? L’équipe a-t-elle l’habitude des rétros ?”
#- Trame: “5 min check-in → 10 min faits marquants → 10 min ce qui aide/ce qui bloque → 10 min idées → 10 min plan d’action (1–2 engagements, responsables, échéance).”
#- Outil: “Icebreaker météo projet (soleil/nuage/pluie).”
#- Prochaine étape: “Planifie le suivi dans 2 semaines avec 1 indicateur simple (ex. délai de traitement des demandes).”
#- Tu veux que je t’envoie une fiche minute pour l’animer demain ? <- Question de relance pour encourager l’expérimentation.


## DEFAULT PROMPT ##
PROMPT = """
<informations de contexte trouvées>
{context}
</informations de contexte trouvées>

En t'aidant si besoin du le contexte ci-dessus et de la conversation, réponds au message :

{question}
"""

## CONTEXT FOR NO CONTEXT QUESTION ##
NO_CONTEXT_MEMORY = """Aucun contexte n'est nécessaire, réponds gentillement à l'utilisateur. N'ajoutes pas de sources en fin de réponse. Si besoin, voilà les informations que tu connais ; 
- Tu es un modèle opensource adapté par Etalab pour aider les agents publics, spécialisé en management et méthodes agiles.
- Tu es connecté a des bases de données spécialisées en management et méthodes agiles.
- L'utilisateur peut regarder en haut a gauche s'il trouve un autre agent spécialisé dans la liste qui pourrait l'aider pour sa question si elle n'est pas sur ton sujet a toi.
"""

## PROMPT FOR CONTEXTUALISED SEARCH ##
PROMPT_SEARCH = """
Tu es un assistant qui cherche des documents dans une base de données spécialisée en management et méthodes agiles pour répondre à une question.
Exemples pour t'aider: 
<history>
Comment organiser une rétrospective efficace ?
</history>
réponse attendue : 
rétrospective agile organisation efficace

<history>
Coucou
</history>
réponse attendue : 
no_search

<history>
Quelles sont les bonnes pratiques pour un daily standup ?
</history>
réponse attendue : 
daily standup bonnes pratiques agile

<history>
Comment définir des OKR dans mon équipe ?
</history>
réponse attendue : 
OKR définition équipe management

<history>
Tu sais faire quoi ?
</history>
réponse attendue : 
no_search

<history>
Quel framework agile choisir pour mon projet ?
</history>
réponse attendue : 
framework agile choix projet scrum kanban

En te basant sur cet historique de conversation : 
<history>
{history}
</history>
question de l'utilisateur : {question}
Réponds avec uniquement une recherche pour trouver des documents sur le management et l'agile qui peuvent t'aider à répondre à la dernière question de l'utilisateur.
Réponds uniquement avec la recherche, rien d'autre, sous forme d'une question claire et précise.
Si l'utilisateur parle du modèle lui même, réponds no_search.
Si la question ne concerne pas le management, l'agile, les méthodes de travail ou l'organisation d'équipe, réponds no_search.
Si aucune recherche n'est nécessaire, réponds no_search.
"""

## Prompt for confidence compute ##
PROMPT_CONFIDENCE = """
Tu es un assistant qui évalue la confiance de la réponse d'un assistant.
Voilà un contexte :
{context}
Voilà une question :
{question}
Voilà une réponse :
{answer}
Réponds avec une note entre 0 et 100 pour la confiance de la réponse en se basant sur le contexte et la question posée.
Si la réponse est "Je ne sais pas" et ne contient pas de sources, réponds 100.
Réponds uniquement avec une note entre 0 et 100 et aucun commentaire.
note : 
"""

def format_chunks_to_text(chunk: dict) -> str:
    """Format chunk metadata to human-readable text."""
    title = f"Document : {chunk['chunk']['metadata']['document_name']}"
    content = f"Content : {chunk['chunk']['content']}"
    return f"{title}\n{content}\n"


def confidence_message(
    client, model, context, question, answer
):
    """Calculate confidence score and yield status message if confidence is low."""
    try:
        confidence = client.chat.completions.create(
            model=model,
            stream=False,
            temperature=0.1,
            max_tokens=3,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT_CONFIDENCE.format(
                        context=context, question=question, answer=answer
                    ),
                }
            ],
        )
        confidence_text = confidence.choices[0].message.content
        confidence_match = re.search(r"\b([0-9]|[1-9][0-9]|100)\b", confidence_text)
        confidence = int(confidence_match.group(1)) if confidence_match else 0
        logger.info(f"Confidence score calculated: {confidence}")
        
        # Only yield warning if confidence is below threshold
        if confidence < 80:
            yield {
                "event": {
                    "type": "status",
                    "data": {
                        "description": f"🔴 Indice de confiance faible : {confidence}% — Prenez cette réponse avec précaution.",
                        "done": True,
                        "hidden": False,
                    },
                }
            }
        else:
            return 0
    except Exception as e:
        logger.error(f"Erreur lors de la génération de la note de confiance: {e}")
        return 0

def stream_albert(
    client,
    model,
    max_tokens,
    messages,
    response_collector=None
):
    """Stream Albert API response."""
    logger.info("=== STREAMING ALBERT RESPONSE ===")
    logger.info(f"Model: {model}")

    try:
        chat_response = client.chat.completions.create(
            model=model,
            stream=True,
            temperature=0.2,
            max_tokens=max_tokens,
            messages=messages,
        )

        output = ""
        word_buffer = ""
        for chunk in chat_response:
            try:
                # Check if chunk has valid structure
                choices = getattr(chunk, "choices", None)
                if not choices or not hasattr(choices[0], "delta"):
                    continue

                delta = choices[0].delta
                finish_reason = getattr(choices[0], "finish_reason", None)
                token = getattr(delta, "content", "") if delta else ""

                if token:
                    output += token
                    word_buffer += token
                    # Yield only when a space is detected (i.e., a word is complete)
                    while " " in word_buffer:
                        word, word_buffer = word_buffer.split(" ", 1)
                        # Add the space back to the word
                        word += " "
                        yield {
                            "event": {
                                "type": "message",
                                "data": {
                                    "content": word,
                                    "done": False,
                                },
                            }
                        }
                # At the end, after finish_reason, flush any remaining word_buffer
                if finish_reason is not None:
                    if word_buffer:
                        yield {
                            "event": {
                                "type": "message",
                                "data": {
                                    "content": word_buffer,
                                    "done": False,
                                },
                            }
                        }
                    break

            except Exception as inner_e:
                logger.error(f"Error in streaming chunk: {inner_e}")
                continue
                
                # Exit when generation is complete
                if finish_reason is not None:
                    break

            except Exception as inner_e:
                logger.error(f"Error in streaming chunk: {inner_e}")
                continue

        # Store complete response if collector provided
        if response_collector is not None:
            response_collector.append(output)

        # Signal completion
        yield {
            "event": {
                "type": "message",
                "data": {"content": "", "done": True},
            }
        }
        logger.info("Albert streaming completed successfully")

    except Exception as e:
        logger.error(f"Global API error: {e}")
        yield {
            "event": {
                "type": "chat:message:delta",
                "data": {
                    "content": f"Erreur globale API : {str(e)}",
                    "done": True,
                },
            }
        }

def search_api_albert(
    collection_ids: List[int],
    user_query: str,
    api_url: str,
    api_key: str = "",
    top_k: int = 5,
    rff_k: int = 20,
    method: str = "semantic",
    score_threshold: float = 0,
    web_search: bool = False,
) -> Optional[Dict]:
    """Performs search in Albert API collections."""
    if method == "hybrid":
        score_threshold = 0
    url = f"{api_url}/search"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "collections": collection_ids,
        "rff_k": rff_k,
        "k": top_k,
        "method": method,
        "score_threshold": score_threshold,
        "web_search": web_search,
        "prompt": user_query,
        "additionalProp1": {},
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        # Filter results by score threshold (manual verification)
        results = []
        for result in response.json().get("data", []):
            if result.get("score", 0) >= score_threshold:
                results.append(result)
        return results

    except requests.RequestException as e:
        logger.error(f"HTTP request error: {e}")
        raise
    except ValueError as e:
        logger.error(f"JSON parsing error: {e}")
        raise


def webpage_to_human_readable(page_content):
    """Convert HTML to readable text."""
    soup = BeautifulSoup(page_content, "html.parser")
    
    # Remove non-content elements
    for element in soup(
        ["script", "style", "meta", "link", "noscript", "header", "footer", "aside"]
    ):
        element.decompose()
    
    text = soup.get_text(separator="\n")
    # Clean up whitespace
    cleaned_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return cleaned_text


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


def pipe_rag(
    self,
    body: dict,
    __event_emitter__=None,
    user_message=None,
    model_id=None,
    messages=None,
    collection_dict: dict = None,
    SYSTEM_PROMPT: str = None,
    PROMPT: str = None,
    format_chunks_to_text: callable = None,
):
    """Main RAG pipeline function."""
    prompt = body["messages"][-1]["content"]

    logger.info("=== PIPE RAG STARTED ===")
    
    # Load configuration
    ALBERT_API_URL = self.valves.ALBERT_API_URL
    ALBERT_API_KEY = self.valves.ALBERT_API_KEY 

    model = self.valves.MODEL
    rerank_model = self.valves.RERANK_MODEL
    number_of_chunks = self.valves.NUMBER_OF_CHUNKS
    number_of_chunks_reranker = self.valves.NUMBER_OF_CHUNKS_RERANKER
    max_tokens = 4096

    user_query = body.get("messages", [])[-1]["content"]

    client = OpenAI(
        api_key=ALBERT_API_KEY,
        base_url=ALBERT_API_URL,
    )
    logger.info("Configuration loaded successfully")
    
    # Prepare messages: System prompt + conversation history
    messages = []
    messages.append(
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    )

    # Add conversation history with rolling window
    history = body.get("messages", [])
    max_hist = self.valves.max_turns
    if history and max_hist > 0:
        recent_history = history[-max_hist:]
        # Find first user message to ensure proper conversation flow
        start_idx = 0
        for i, msg in enumerate(recent_history):
            if msg.get("role") == "user":
                start_idx = i
                break
        messages += recent_history[start_idx:]

    # Search for relevant documents
    try:
        # Generate search query using LLM
        search = client.chat.completions.create(
            model=model,
            stream=False,
            temperature=0.1,
            max_tokens=50,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT_SEARCH.format(
                        history=messages[1:],
                        question=prompt,
                    ),
                }
            ],
        )
        logger.debug(f"Collection dict: {collection_dict}")
        logger.debug(f"Search messages: {[msg for msg in messages[1:]]}")

        search = search.choices[0].message.content
        logger.info(f"Generated search query: {search}")
        
        if search.strip().lower() == "no_search":
            top_chunks = []
            context = NO_CONTEXT_MEMORY
        else:
            yield {
                "event": {
                    "type": "status",
                    "data": {
                        "description": f"Recherche en cours pour '{search}'",
                        "done": False,
                        "hidden": False,
                    },
                }
            }

            # Handle internet search vs collection search
            #if list(collection_dict.keys()) == ["internet"] or ("internet" in search.strip().lower() and PROPOSE_NET):
            #    logger.info("Performing internet search")
            #    context = search_internet(ALBERT_API_URL, ALBERT_API_KEY, search)
            #    context = ''
            #    logger.debug(f"Internet search context retrieved: \n\n\n{context}\n\n\n")
            #    context = f"Voila ce qui a été trouvé sur internet : {context} Réponds à l'utilisateur en utilisant ce qui a été trouvé."
            #else:
            logger.info("Performing Albert API search")
            top_chunks = search_api_albert(
                collection_ids=list(collection_dict.values()),
                user_query=search,
                api_url=ALBERT_API_URL,
                api_key=ALBERT_API_KEY,
                top_k=number_of_chunks,
                rff_k=number_of_chunks,
                method=self.valves.SEARCH_METHOD,
                score_threshold=self.valves.SEARCH_SCORE_THRESHOLD,
                web_search=False,
            )
            logger.info(f"Search results found: {len(top_chunks)}")
            
            # Rerank results for better relevance
            #logger.info("Starting reranking process")
            if rerank_model != "None":
                top_chunks = reranker(
                    query=user_query,
                    chunks=top_chunks,
                    api_url=ALBERT_API_URL,
                    api_key=ALBERT_API_KEY,
                    score_threshold=self.valves.RERANKER_SCORE_THRESHOLD,
                    rerank_model=rerank_model,
                    min_chunks=number_of_chunks,
                )[:number_of_chunks_reranker]

            # Format context from chunks
            references = ""
            for k, chunk in enumerate(top_chunks):
                references += f"""
- Document {k+1}:
{format_chunks_to_text(chunk = chunk)}
"""
            logger.debug(f"Context generated: {len(references)} characters")
            context = references

    except Exception as e:
        logger.error(f"Error during search process: {e}")
        yield {
            "event": {
                "type": "status",
                "data": {
                    "description": "Erreur lors de la recherche.",
                    "done": True,
                    "hidden": False,
                },
            }
        }
        return "Désolé, on dirait que la connection à AlbertAPI est perdue. Veuillez réessayer plus tard."

    yield {
        "event": {
            "type": "status",
            "data": {
                "description": "Terminé.",
                "done": True,
                "hidden": True,
            },
        }
    }

    # Add user query with context to messages
    messages[-1] = {
            "role": "user",
            "content": PROMPT.format(context=context, question=prompt),
        }

    logger.debug("Messages prepared for Albert response generation")
    [logger.info(f"Message: {str(msg)[:50]}...") for msg in messages]

    # Generate and stream response
    answer = ""
    for resp in stream_albert(client, model, max_tokens, messages, __event_emitter__):
        answer += resp["event"]["data"]["content"]
        yield resp

    logger.info(f"Final answer generated: {answer[:100]}...")
    
    # Calculate and display confidence if needed
    for conf_event in confidence_message(client, model, context, prompt, answer):
        yield conf_event

    # Add assistant response to conversation
    body["messages"].append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
    return body


class Pipeline:
    class Valves(BaseModel):
        max_turns: int = Field(
            default=5, description="Maximum conversation turns taken into account for a user."
        )

        ALBERT_API_URL: str = Field(default="https://albert.api.etalab.gouv.fr/v1")
        ALBERT_API_KEY: str = Field(default="")
        MODEL: Literal[
            "albert-small",
            "albert-large",
        ] = Field(default="albert-large")
        RERANK_MODEL: Literal["BAAI/bge-reranker-v2-m3", "None"] = Field(default="BAAI/bge-reranker-v2-m3")
        NUMBER_OF_CHUNKS: int = Field(default=20)
        NUMBER_OF_CHUNKS_RERANKER: int = Field(default=10)
        SEARCH_SCORE_THRESHOLD: float = Field(default=0, description="Score threshold for the search API. 0 if method is hybrid.")
        SEARCH_METHOD: Literal["semantic", "hybrid"] = Field(default="hybrid", description="Method for the search API. semantic if method is hybrid.")
        RERANKER_SCORE_THRESHOLD: float = Field(default=0.1)
        pass
 
    def __init__(self):
        self.name = PIPELINE_NAME
        self.valves = self.Valves()
        self.collection_dict = collection_dict
        self.SYSTEM_PROMPT = SYSTEM_PROMPT
        self.PROMPT = PROMPT
        if self.valves.SEARCH_METHOD == "hybrid":
            self.valves.SEARCH_SCORE_THRESHOLD = 0

    async def on_startup(self):
        """Called when the server is started."""
        logger.info(f"Pipeline startup: {__name__}")

    async def on_shutdown(self):
        """Called when the server is stopped."""
        logger.info(f"Pipeline shutdown: {__name__}")

    def pipe(
        self,
        body: dict,
        __event_emitter__=None,
        user_message=None,
        model_id=None,
        messages=None,
    ):
        """Main pipeline entry point."""
        return pipe_rag(
            self,
            body,
            __event_emitter__,
            user_message,
            model_id,
            messages,
            self.collection_dict,
            self.SYSTEM_PROMPT,
            self.PROMPT,
            format_chunks_to_text,
        )
