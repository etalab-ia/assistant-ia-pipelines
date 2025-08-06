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

PIPELINE_NAME = "Assistant Service Public"
collection_dict = {"travail-emploi": 784, "service-public": 785}
PROPOSE_NET = True

if PROPOSE_NET:
    manque_info = "propose à l'utilisateur que tu cherches sur internet si tu n'a pas déjà proposé."
    exemple_net = """<history>
Comment faire un pret pour un appartement ?
Assistant : Voulez vous chercher sur internet ?
User : Oui
</history>
réponse attendue : 
internet faire un pret pour un appartement
Si l'utilisateur dis oui a une demande de recherche sur internet, commences ta recherche par internet suivi de la recherche."""
    if_net = "et internet"
else:
    manque_info = "demande des précisions à l'utilisateur"
    exemple_net = ""
    if_net = ""

#### PROMPTS ####
## SYSTEM PROMPT ##
SYSTEM_PROMPT = """
Tu es """ + PIPELINE_NAME + """, un assistant développé par l'Etalab qui répond à des questions en te basant sur un contexte.
Tu parles en français. Tu es précis et poli.
Tu es connecté aux collections suivantes : {collections} sur AlbertAPI """+ if_net + """.
Ce que tu sais faire : Tu sais répondre aux questions et chercher dans les bases de connaissance de Albert API """+ if_net + """.
Pour les questions sur des sujets spécifiques autres que tes connaissances, invites l'utilisateur à se tourner vers un autre assistant spécialisé.
Ne donnes pas de sources si tu réponds à une question meta ou sur toi.
"""

## DEFAULT PROMPT ##
PROMPT = """
<context trouvé>
{context}
</context trouvé>

En t'aidant si besoin du le contexte ci-dessus et de la conversation, réponds au dernier message de l'utilisateur :
<question>
{question}
</question>

Continue la conversation de manière claire.
A la fin de ta réponse, ajoute les sources ou liens urls utilisés pour répondre à la question. Quand tu mets des liens donne leurs des noms simples avec la notation markdown.
Si tu mets des sources en fin de réponse, ne mets QUE les sources liées à ta réponse, jamais de source inutilement.
Si tu ne trouves pas d'éléments de réponse dans le contexte ou dans ton prompt system, réponds que tu manques d'informations ou """ + manque_info + """. Sois poli.
"""

## CONTEXT FOR NO CONTEXT QUESTION ##
NO_CONTEXT_MEMORY = """Aucun contexte n'est nécessaire, réponds gentillement à l'utilisateur. N'ajoutes pas de sources en fin de réponse. Si besoin, voilà les informations que tu connais ; 
- Tu es un modèle opensource adapté par Etalab pour aider les agents publics.
- Tu es connecté a des bases de données.
- L'utilisateur peut regarder en haut a gauche s'il trouve un autre agent spécialisé dans la liste qui pourrait l'aider pour sa question si elle n'est pas sur ton sujet a toi.
"""

## PROMPT FOR CONTEXTUALISED SEARCH ##
PROMPT_SEARCH = """
Tu es un assistant qui cherche des documents pour répondre à une question.
Exemples pour t'aider: 
<history>
Ma soeur va se marier, j'ai le droit a des jours de congés ?
</history>
réponse attenue : 
jours de congés pour mariage frere ou soeur

<history>
Coucou
</history>
réponse attenue : 
no_search

<history>
Où travaille John Doe ?
</history>
réponse attenue : 
John Doe travail

<history>
Qui est le CEO de blackrock ?
</history>
réponse attenue : 
CEO Blackrock

<history>
Tu sais faire quoi ?
<history>
réponse attendue : 
no_search

""" + exemple_net + """


En te basant sur cet historique de conversation : 
<history>
{history}
</history>
question de l'utilisateur : {question}
Réponds avec uniquement une recherche pour trouver des documents qui peuvent t'aider à répondre à la dernière question de l'utilisateur.
Réponds uniquement avec la recherche, rien d'autre. Donnes entre 2 et 10 mots clés pertinents.
Si l'utilisateur parle du modèle lui même, réponds no_search.
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
    urls_to_concatenate = f"URL de la fiche pratique : {chunk.get('url', '')}"
    title = f"Titre : {chunk.get('title', '')}"
    context = "Contexte : "
    for context_item in chunk.get("context", []):
        context += f"{context_item}, "
    text = f"Texte : \n{chunk.get('description','')}\n{chunk.get('text_content', '')}"

    fields = [
        urls_to_concatenate if urls_to_concatenate else "",
        title if title else "",
        context if context else "",
        text if text else "",
    ]
    final_text = ".\n".join([f for f in fields if f]).strip()

    return final_text


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
                        "description": f"🔴 Indice de confiance faible : {confidence}% — Vérifiez les sources.",
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
                    yield {
                        "event": {
                            "type": "message",
                            "data": {
                                "content": token,
                                "done": False,
                            },
                        }
                    }
                
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


def search_internet(api_url, api_key, prompt):
    """Search internet using Albert API."""
    session = requests.session()
    session.headers = {"Authorization": f"Bearer {api_key}"}

    data = {"collections": [], "web_search": True, "k": 10, "prompt": prompt}
    response = session.post(url=f"{api_url}/search", json=data)

    logger.debug("Internet search results retrieved")

    chunks = [
        f"Context {n} : {webpage_to_human_readable(result['chunk']['content'])}"
        for n, result in enumerate(response.json()["data"])
    ]
    sources = list(
        set(
            [
                result["chunk"]["metadata"]["document_name"]
                for result in response.json()["data"]
            ]
        )
    )

    context = "\n\n".join(chunks) + "\n\nSources : " + ", ".join(sources)

    if len(context.strip()) < 100 : 
        context = "Rien de pertinent n'a été trouvé sur internet, avertissez l'utilisateur que vous êtes bloqué par une whitelist.."

    return context


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
            "content": SYSTEM_PROMPT.format(collections=collection_dict.keys()),
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
            if list(collection_dict.keys()) == ["internet"] or ("internet" in search.strip().lower() and PROPOSE_NET):
                logger.info("Performing internet search")
                context = search_internet(ALBERT_API_URL, ALBERT_API_KEY, search)
                logger.debug(f"Internet search context retrieved: \n\n\n{context}\n\n\n")
                context = f"Voila ce qui a été trouvé sur internet : {context} Réponds à l'utilisateur en utilisant ce qui a été trouvé."
            else:
                logger.info("Performing Albert API search")
                search_results = search_api_albert(
                    collection_ids=list(collection_dict.values()),
                    user_query=search,
                    api_url=ALBERT_API_URL,
                    api_key=ALBERT_API_KEY,
                    top_k=20,
                    rff_k=20,
                    method="semantic",
                    score_threshold=self.valves.SEARCH_SCORE_THRESHOLD,
                    web_search=False,
                )
                logger.info(f"Search results found: {len(search_results)}")
                
                # Rerank results for better relevance
                logger.info("Starting reranking process")
                top_chunks = reranker(
                    query=user_query,
                    chunks=search_results,
                    api_url=ALBERT_API_URL,
                    api_key=ALBERT_API_KEY,
                    score_threshold=self.valves.RERANKER_SCORE_THRESHOLD,
                    rerank_model=rerank_model,
                    min_chunks=number_of_chunks,
                )[:number_of_chunks]

                # Format context from chunks
                references = ""
                for k, chunk in enumerate(top_chunks):
                    references += f"""
    - Document {k+1}:
    {format_chunks_to_text(chunk = chunk.get("chunk").get("metadata"))}
    """
                logger.info(f"Reranked results: {len(top_chunks)} chunks")
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
        RERANK_MODEL: str = Field(default="BAAI/bge-reranker-v2-m3")
        NUMBER_OF_CHUNKS: int = Field(default=5)
        SEARCH_SCORE_THRESHOLD: float = Field(default=0.35)
        RERANKER_SCORE_THRESHOLD: float = Field(default=0.1)
        pass
 
    def __init__(self):
        self.name = PIPELINE_NAME
        self.valves = self.Valves()
        self.collection_dict = collection_dict
        self.SYSTEM_PROMPT = SYSTEM_PROMPT
        self.PROMPT = PROMPT

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
