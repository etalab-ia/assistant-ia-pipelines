import logging
import os
import re
import requests
import time
from bs4 import BeautifulSoup
from openai import OpenAI
from typing import Iterator, Union, Generator

from pydantic import BaseModel, Field
from typing import Literal

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mon_logger")
logger.setLevel(logging.DEBUG)

PIPELINE_NAME="Assistant Service Public – Avec raisonnement"
goggles = []  # ["https://gist.github.com/camilleAND/751a30edc9ae6304b345750ff77ba0dd"] #For BRAVE goggles
collections_ids = [784,785] #Ex : [785] If empty, web search will be used



#### PROMPTS ####

SYSTEM_PROMPT = f"""
Tu es un assistant qui répond à des questions en te basant sur un contexte. Tu parles en français. Tu es précis et poli.
{"Tu as accès a un RAG." if collections_ids else "Tu as accès a Internet."}
"""

PROMPT_SEARCH = """
Tu es un assistant qui cherche des documents pour répondre à une question. Tu te bases sur le contexte de la conversation pour comprendre les questions à poser.
{prompt_search_addon}
Exemples pour t'aider: 
<history>
Ma soeur va se marier, j'ai le droit a des jours de congés ?
</history>
réponse attenue : 
Peut-on avoir des jours de congés pour le mariage de sa soeur ?

<history>
Tu sais faire quoi ?
</history>
réponse attenue : 
no_search

<history>
Où travaille John Doe ?
</history>
réponse attenue : 
Quel est le lieu de travail de John Doe ?

En te basant sur cet historique de conversation : 
<history>
{history}
</history>
question de l'utilisateur : {question}
Réponds avec uniquement une ou des questions pour trouver des documents qui peuvent t'aider à répondre à la dernière question de l'utilisateur.
Réponds uniquement avec la recherche, rien d'autre. La recherche dois être une question précise et auto suffisante.
Si aucune recherche n'est nécessaire, réponds "no_search".
"""

PROMPT_COMPLEXE_OR_NOT = """
Tu es un assistant qui détermine si une question est complexe ou non.
Tu te bases sur le contexte de la conversation pour comprendre la question.
{prompt_complex_or_not_addon}

Exemples pour t'aider: 
<history>
assistant : Bonjour, comment je peux t'aider ?
utilisateur : Ma soeur va se marier, j'ai le droit a des jours de congés ?
</history>
réponse attenue : 
easy

<history>
assistant : Bonjour, comment je peux t'aider ?
utilisateur : Où travaille John Doe ?
</history>
réponse attenue : 
easy

<history>
assistant : Bonjour, comment je peux t'aider ?
utilisateur : Donnes moi toutes les preuves de l'existence de la matière noire.
</history>
réponse attenue : 
complex

<history>
assistant : Bonjour, comment je peux t'aider ?
utilisateur : Quel est le prix du litre d'essence ?
</history>
réponse attenue : 
easy

<history>
assistant : Bonjour, comment je peux t'aider ?
utilisateur : Si je suis handicapé à 80% et né en 1950, combien de trimestres ai-je validés ?
</history>
réponse attenue : 
complex

En te basant sur cet historique de conversation : 
<history>
{history}
</history>
question de l'utilisateur : {question}
Réponds avec "easy" ou "complex" en fonction de la complexité de la question. Ne donnes pas d'explication.
"""


#### END PROMPTS ####


# For the citations
def extract_first_url(text):
    """
    Extracts the first URL found in a string.

    Args:
        text (str): The string to search for URLs

    Returns:
        str or None: The first URL found, or None if no URL is found
    """
    # Regular expression pattern to match URLs
    # This pattern matches http, https, ftp URLs with various domain formats
    url_pattern = re.compile(
        r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
        re.IGNORECASE,
    )

    # Find the first match
    match = url_pattern.search(text)

    if match:
        return match.group(0)  # Return the matched URL
    else:
        return None

def search_api_albert(
    prompt: str, k: int = 5, collection_ids: list = None
) -> list:
    """
    Cet outil permet de chercher des bouts de documents sur le travail et le droit en france.

    Args:
        prompt: les mots clés ou phrases a chercher sémantiquement pour trouver des documents (ex: prompt="procuration vote france")
    """
    logger.info(f"=== ALBERT API SEARCH ===")
    logger.info(f"Searching for: {prompt}")
    logger.info(f"Collections: {collection_ids}")
    
    docs = []
    names = []
    for coll in collection_ids:
        data = {"collections": [coll], "k": k, "prompt": prompt}
        response = requests.post(
            url="https://albert.api.etalab.gouv.fr/v1/search",
            json=data,
            headers={"Authorization": f'Bearer {os.getenv("ALBERT_KEY")}'},
        )
        logger.info(f"Response extract: {response.text[:100]}...")
        docs_coll = []
        for result in response.json()["data"]:
            content = result["chunk"]["content"]
            if len(content) < 150:
                continue
            name = result["chunk"]["metadata"]["document_name"]
            names.append(name)
            score = result["score"]
            metadata_dict = result["chunk"]["metadata"]
            source = " - ".join(
                [
                    f"{metadata_dict[stuff]}"
                    for stuff in metadata_dict
                    if stuff in ["titre", "title", "client", "url"]
                ]
            )
            docs_coll.append((content, name, source, score))
        docs = docs + docs_coll
    docs = sorted(docs, key=lambda x: x[3], reverse=True)
    docs = [
        f"[{name}] {doc[2]} {doc[0].split('Extrait article :')[-1]}"
        for name, doc in zip(names, docs)
    ]
    
    logger.info(f"Found {len(docs)} relevant documents")
    return docs[:k]

class Prompts:
    @staticmethod
    def researcher(num_queries):
        return f"""Tu es un assistant expert en recherche. En te basant sur la demande utilisateur, génère jusqu'à {num_queries} distinctes
        différentes et simples queries google (comme un humain le ferait) qui aideraient à recueillir des informations sur le sujet demandé. Dans tes queries google mets aussi les mots clés présents dans la question de l'utilisateur.
        Si l'utilisateur ne précise pas son pays, part du principe qu'il est Français et que sa demande est en France.
        Réponds uniquement avec une liste python, par exemple : ["query1", "query2"] et ne dis rien d'autre. Les queries google ne doivent pas se ressembler."""

    @staticmethod
    def evaluator():
        return """Vous êtes un évaluateur critique de recherche. Étant donné la requête de l'utilisateur et le contenu d'une page web,
        déterminez si la page web contient des informations utiles pour répondre à la requête. Vous ne voyez ici qu'un extrait de la page.
        Répondez avec exactement un mot : 'oui' si la page est utile ou en lien avec la requête, ou 'non' si elle ne l'est pas ou n'a pas l'air utile. N'incluez aucun texte supplémentaire"""

    @staticmethod
    def extractor():
        return """Tu es un expert en extraction d'information, en te basant sur la demande utilisateur qui a amené à cette page, et son contenu, extrait et résume toutes les informations qui pourraient aider à répondre à la demande utilisateur.
        Réponds uniquement avec le résumé du contexte pertinent sans commentaire supplémentaire. Ne gardes que ce qui est en lien avec la requête utilisateur. Donnes aussi le title des articles et les URLs complètes dans ta réponse en commençant la réponse par 'Selon [titre ou url].' quand c'est possible. 
        Elimine tous les articles qui ne parle pas de choses intéressantes pour la question de l'utilisateur. Si rien n'est intéressant pour l'utilisateur, réponds <suivant>."""

    @staticmethod
    def analytics():
        return """Vous êtes un assistant de recherche analytique. Sur la base de la requête initiale, des recherches effectuées jusqu'à présent et des contextes extraits des pages web, déterminez si des recherches supplémentaires sont nécessaires. 
        Si le contexte permet de répondre à l'utilisateur, répondez []. Ne fais pas de recherches inutiles.
        Si les contextes extraits sont vides ou si des recherches supplémentaires sont absolument nécessaires, fournissez jusqu'à deux nouvelles requêtes de recherche sous forme de liste Python (par exemple, ["new query1", "new query2"]). Si aucune recherche supplémentaire n'est nécessaire répondez uniquement avec une liste vide []. N'affichez qu'une liste Python ou une liste vide[] sans aucun texte supplémentaire.
        Ne fais jamais de recherches supplémentaires si le contexte est suffisant pour répondre à la question.
        """

    @staticmethod
    def redactor():
        return """Vous êtes un expert en rédaction de réponses de demande utilisateur. Soyez poli.
        Sur la base des contextes rassemblés ci-dessus et de la requête initiale, rédigez une réponse complete, 
        bien structurée en markdown et détaillée qui répond à la question de manière approfondie. Ne faites pas d'introduction, commencez tout de suite avec la réponse. Incluez des références sous la forme '[numero reference]' dans les paragraphes que vous rédigez qui renvoient aux références utilisées.
        Mettez les références en bas de la réponse s'il vous en citez. N'utilisez que les références disponibles.
        Incluez toutes les informations et conclusions utiles sans commentaires supplémentaires, ainsi que les noms d'articles et urls présents dans le contexte qui semble pertinents. Dans les références veillez a ne pas faire de doublons priorisez les URLs et les titres. N'utilises que les urls qui sont dans le contexte rassemblé.
        """


class SyncHelper:
    @staticmethod
    def call_openrouter(
        client, messages, max_tokens=2048, step=False
    ):
        model = {
            "END": os.getenv("REDACTOR_MODEL"),
            "ANALYTICS": os.getenv("ANALYTICS_MODEL"),
        }.get(step, os.getenv("DEFAULT_MODEL"))

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }
        try:
            resp = client.chat.completions.create(**payload)
            answer = resp.choices[0].message.content
            return answer
        except Exception as e:
            logger.error(f"Error calling OpenRouter: {e}")
            return None

    @staticmethod
    def generate_search_queries(
        client, user_query, num_queries=2
    ):
        logger.info(f"=== GENERATING SEARCH QUERIES ===")
        logger.info(f"User query: {user_query}")
        logger.info(f"Number of queries to generate: {num_queries}")
        
        prompt = Prompts.researcher(num_queries)
        messages = [
            {
                "role": "system",
                "content": (
                    "Vous êtes un assistant de recherche précis et utile."
                ),
            },
            {
                "role": "user",
                "content": f"Demande utilisateur: {user_query}\n\n{prompt}",
            },
        ]
        response = SyncHelper.call_openrouter(
            client, messages, max_tokens=150
        )
        if response:
            try:
                search_queries = eval(response)
                if isinstance(search_queries, list):
                    logger.info(f"Generated search queries: {search_queries}")
                    return search_queries
                else:
                    logger.error(f"LLM did not return a list. Response: {response}")
                    return []
            except Exception as e:
                logger.error(f"Error parsing search queries: {e}. Response: {response}")
                return []
        return []

    @staticmethod
    def perform_brave_search(query, k=2, num_queries=2):
        logger.info(f"=== BRAVE SEARCH ===")
        logger.info(f"Query: {query}")
        
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": os.getenv("BRAVE_KEY"),
        }
        params = {
            "q": query,
            "search_lang": "fr",
            "count": k,
            "country": "fr",
            "goggles": goggles,
        }
        try:
            resp = requests.get(url, headers=headers, params=params)
            if resp:
                results = resp.json()
                links = [
                    result.get("url")
                    for result in results.get("web", {}).get("results", [])
                ]
                logger.info(f"Found {len(links)} links from Brave search")
                if num_queries > 1:
                    time.sleep(1)
                return links[:k]
            else:
                text = resp.text
                logger.error(f"Brave search API error: {resp.status_code} - {text}")
                return []
        except Exception as e:
            logger.error(f"Error performing Brave search: {e}")
            return []

    @staticmethod
    def webpage_to_human_readable(page_content):
        soup = BeautifulSoup(page_content, "html.parser")
        # Remove non-content elements
        for element in soup(
            ["script", "style", "meta", "link", "noscript", "header", "footer", "aside"]
        ):
            element.decompose()
        text = soup.get_text(separator="\n")
        # Clean up whitespace
        cleaned_text = "\n".join(
            line.strip() for line in text.splitlines() if line.strip()
        )

        logger.debug(f"Cleaned text preview: {cleaned_text[:100]}...")
        return cleaned_text

    @staticmethod
    def fetch_webpage_text(url):
        logger.info(f"=== FETCHING WEBPAGE ===")
        logger.info(f"URL: {url}")
        
        try:
            headers = {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
            }
            resp = requests.get(url, headers=headers, timeout=10)
            if resp:
                page_content = resp.text
                cleaned_content = SyncHelper.webpage_to_human_readable(page_content)
                logger.info(f"Successfully fetched content from {url}")
                return cleaned_content
            else:
                logger.error(f"Fetch error for {url}: {resp}")
                return ""
        except Exception as e:
            logger.error(f"Error fetching webpage text for {url}: {e}")
            return ""

    @staticmethod
    def is_page_useful(
        client, user_query, page_text
    ):
        if not page_text:
            return "Non"

        prompt = Prompts.evaluator()
        messages = [
            {
                "role": "system",
                "content": (
                    "Vous êtes un évaluateur strict et précis de la pertinence des recherches."
                ),
            },
            {
                "role": "user",
                "content": f"Requête utilisateur: {user_query}\n\nExtrait de page web (premiers 5000 caractères) :\n{page_text[:5000]}[...]\n\n{prompt}",
            },
        ]

        response = SyncHelper.call_openrouter(
            client, messages, max_tokens=10
        )
        if response:
            answer = response.strip().lower()
            logger.debug(f"Page usefulness evaluation: {answer}")
            return "Oui" if "oui" in answer or "yes" in answer else "Non"
        return "Non"

    @staticmethod
    def extract_relevant_context(
        client,
        user_query,
        search_query,
        page_text,
        max_tokens=1024,
    ):
        if not page_text:
            return ""

        logger.info(f"=== EXTRACTING CONTEXT ===")
        logger.info(f"User query: {user_query}")
        logger.info(f"Search query: {search_query}")

        prompt = Prompts.extractor()
        messages = [
            {
                "role": "system",
                "content": (
                    "Vous êtes un expert dans l'extraction et la synthèse d'informations."
                ),
            },
            {
                "role": "user",
                "content": f"Requête utilisateur: {user_query}\nRequête de recherche: {search_query}\n\nContexte trouvé (premiers 20000 caractères) :\n{page_text[:20000]}\n\n{prompt}",
            },
        ]
        response = SyncHelper.call_openrouter(
            client, messages, max_tokens=max_tokens
        )
        if response:
            logger.info("Context extraction completed successfully")
            return response.strip()
        return ""

    @staticmethod
    def get_new_search_queries(
        client,
        user_query,
        previous_search_queries,
        all_contexts,
    ):
        logger.info(f"=== EVALUATING NEED FOR ADDITIONAL SEARCHES ===")
        logger.info(f"Previous queries: {previous_search_queries}")
        logger.info(f"Contexts found: {len(all_contexts)}")
        
        if not all_contexts:
            # If no contexts found, generate new queries directly
            return SyncHelper.generate_search_queries(
                client, user_query, 2
            )

        context_combined = "\n".join(
            [f"{context[:1000]} [...]" for context in all_contexts]
        )
        prompt = Prompts.analytics()
        messages = [
            {
                "role": "system",
                "content": (
                    "Vous êtes un planificateur de recherche systématique."
                ),
            },
            {
                "role": "user",
                "content": f"Contexte pertinent trouvé:\n{context_combined}\n\n{prompt}\nDemande utilisateur: {user_query}\nRecherches précédentes déjà effectuées: {previous_search_queries}",
            },
        ]
        response = SyncHelper.call_openrouter(
            client, messages, max_tokens=100, step="ANALYTICS"
        )
        if response:
            cleaned = response.strip()
            logger.info(f"Analytics evaluation result: {cleaned}")
            if "[]" in cleaned:
                logger.info("No additional research needed")
                return "[]"
            try:
                new_queries = eval(cleaned)
                if isinstance(new_queries, list):
                    logger.info(f"New search queries generated: {new_queries}")
                    return new_queries
                else:
                    logger.error(f"LLM did not return a list for new search queries. Response: {response}")
                    return []
            except Exception as e:
                logger.error(f"Error parsing new search queries: {e}. Response: {response}")
                return []
        return []

    @staticmethod
    def generate_final_prompt(
        client,
        user_query,
        all_contexts,
        urls,
        prompt_suffix,
        max_tokens,
    ):
        logger.info(f"=== GENERATING FINAL PROMPT ===")
        logger.info(f"Number of contexts: {len(all_contexts)}")
        
        if not all_contexts:
            return "No relevant information found to answer your query."

        if prompt_suffix:
            prompt_suffix = f"User instructions importantes: {prompt_suffix}"
        else:
            prompt_suffix = ""
        context_combined = "\n".join(all_contexts)
        prompt = Prompts.redactor()
        prompt = f"Contextes pertinents rassemblés:\n{context_combined}\n{urls}\n\n{prompt}\n{prompt_suffix}\n\nDemande utilisateur: {user_query}"
        
        logger.info("Final prompt generated successfully")
        return prompt

    @staticmethod
    def process_link(
        client,
        link,
        user_query,
        search_query,
        log,
        num_queries=2,
    ):
        logger.info(f"=== PROCESSING LINK ===")
        logger.info(f"Link: {link}")
        
        log.append(f"Fetching content from: {link}")
        page_text = SyncHelper.fetch_webpage_text(link)
        if not page_text:
            log.append(f"Failed to fetch content from: {link}")
            logger.warning(f"Failed to fetch content from: {link}")
            return ""
        if num_queries > 1:
            usefulness = SyncHelper.is_page_useful(
                client, user_query, page_text
            )
            log.append(f"Page usefulness for {link}: {usefulness}")
            logger.info(f"Page usefulness evaluation: {usefulness}")
            if usefulness == "Oui":
                context = SyncHelper.extract_relevant_context(
                    client,
                    user_query,
                    search_query,
                    page_text,
                )
                if context:
                    logger.info(f"Context extracted successfully from {link}")
                    log.append(
                        f"Extracted context from {link} (first 200 chars): {context[:200]}"
                    )
                    return f"\n<{link}>\n{context}\n</{link}>"
        elif num_queries == 1:
            logger.info(f"Simple search mode - using page text directly from {link}")
            return str([link]) + page_text
        return ""

    @staticmethod
    def process_query_api(
        client,
        user_query,
        search,
        k,
        log,
        collections_ids=None,
    ):
        logger.info(f"=== PROCESSING API QUERY ===")
        logger.info(f"Search query: {search}")
        logger.info(f"Collections: {collections_ids}")
        
        log.append(f"Fetching content from: {user_query}")
        try:
            docs = search_api_albert(search, k, collections_ids)

            if not docs:
                logger.warning(f"No documents found for: {search}")
                log.append(f"No documents found for: {search}")
                return []

            useful_docs = []
            for doc in docs:
                usefulness = SyncHelper.is_page_useful(
                    client, user_query, doc
                )
                if usefulness == "Oui":
                    useful_docs.append(doc)
            
            logger.info(f"Found {len(useful_docs)} useful documents out of {len(docs)} total")
            log.append(f"Number of useful docs: {len(useful_docs)}")

            return useful_docs
        except Exception as e:
            logger.error(f"Error in process_query_api: {e}")
            log.append(f"Error in process_query_api: {e}")
            return []

    @staticmethod
    def process_useful_api_docs(
        client, user_query, useful_docs, log
    ):
        logger.info(f"=== PROCESSING USEFUL API DOCUMENTS ===")
        logger.info(f"Number of documents to process: {len(useful_docs)}")
        
        if not useful_docs:
            return []

        extracted_contexts = []
        for doc in useful_docs:
            context = SyncHelper.extract_relevant_context(
                client,
                user_query,
                user_query,
                doc,
                max_tokens=512,
            )
            if context and context.strip() and context.lower() != "<suivant>":
                extracted_contexts.append(context)

        logger.info(f"Successfully extracted {len(extracted_contexts)} contexts from API documents")
        for context in extracted_contexts:
            log.append(f"Extracted context (first 200 chars): {context[:200]}")

        return extracted_contexts


def emit_status(event_emitter, description):
    if event_emitter:
        event_emitter({
            "type": "status",
            "data": {"description": description, "done": False, "hidden": False},
        })


def sync_research(
    client,
    user_query,
    internet,
    iteration_limit,
    prompt_suffix=None,
    max_tokens=1024,
    num_queries=2,
    k=5,
    collections_ids=None,
):
    logger.info(f"=== STARTING RESEARCH SESSION ===")
    logger.info(f"User query: {user_query}")
    logger.info(f"Internet mode: {internet}")
    logger.info(f"Iteration limit: {iteration_limit}")
    logger.info(f"Number of queries: {num_queries}")

    search_type = "internet" if internet else "le RAG"
    search_type_description = f"Je lance une recherche approfondie sur {search_type}" if num_queries > 1 else f"Je lance une recherche rapide sur {search_type}"
    
    start_time = time.time()
    aggregated_contexts = []
    aggregated_chunks = []
    all_search_queries = []
    log_messages = []
    iteration = 0

    # Input validation
    if not user_query or not user_query.strip():
        logger.error("Empty query provided")
        yield "Please provide a valid query."
        return

    iteration_limit = max(1, min(iteration_limit, 10))  # Ensure between 1 and 10
    num_queries = max(1, min(num_queries, 5))  # Ensure between 1 and 5
    k = max(1, min(k, 10))  # Ensure between 1 and 10

    try:
        log_messages.append(f"Starting research for: {user_query}")
        log_messages.append("Generating initial search queries...")
        yield {
            "event": {
                "type": "status",
                "data": {"description": f"{search_type_description}...", "done": False, "hidden": False},
            }
        }
        time.sleep(2)

        new_search_queries = SyncHelper.generate_search_queries(
            client, user_query, num_queries
        )

        if not new_search_queries:
            log_messages.append(
                "No search queries were generated. Using the original query."
            )
            new_search_queries = [user_query]

        all_search_queries.extend(new_search_queries)
        log_messages.append(f"Initial search queries: {new_search_queries}")
        yield {
            "event": {
                "type": "status",
                "data": {"description": f"Je cherche {', '.join(new_search_queries)}...", "done": False, "hidden": False},
            }
        }

        while iteration < iteration_limit:
            logger.info(f"=== ITERATION {iteration + 1} ===")
            log_messages.append(f"\n=== Iteration {iteration + 1} ===")
            iteration_contexts = []
            all_links = []

            if internet:
                yield {
                    "event": {
                        "type": "status",
                        "data": {"description": "Je me connecte à internet...", "done": False, "hidden": False},
                    }
                }
                logger.info("Launching internet search")
                search_results = []
                for query in new_search_queries[:num_queries]:
                    search_results.append(
                        SyncHelper.perform_brave_search(
                            query, k, num_queries
                        )
                    )

                unique_links = {}
                for idx, links in enumerate(search_results):
                    query_used = new_search_queries[idx]
                    for link in links:
                        if link not in unique_links:
                            unique_links[link] = query_used

                logger.info(f"Found {len(unique_links)} unique links in iteration {iteration + 1}")
                log_messages.append(
                    f"Found {len(unique_links)} unique links in iteration {iteration + 1}."
                )

                if not unique_links:
                    log_messages.append("No links found in this iteration.")
                else:
                    link_results = []
                    for link in unique_links:
                        yield {
                            "event": {
                                "type": "status",
                                "data": {"description": f"Je lis le contenu de {link}...", "done": False, "hidden": False},
                            }
                        }
                        result = SyncHelper.process_link(
                            client,
                            link,
                            user_query,
                            unique_links[link],
                            log_messages,
                            num_queries,
                        )
                        if result:
                            link_results.append(result)
                            yield {
                                "event": {
                                    "type": "status",
                                    "data": {"description": f"J'ai extrait le contenu pertinent de {link}...", "done": False, "hidden": False},
                                }
                            }
                    iteration_contexts = link_results
                    aggregated_chunks.extend([link for link in unique_links])
                    all_links.extend([link for link in unique_links])
            else:
                logger.info("Launching RAG search")
                if num_queries > 1:
                    yield {
                        "event": {
                            "type": "status",
                            "data": {"description": "Je lance une recherche approndie dans les bases de connaissances disponibles...", "done": False, "hidden": False},
                        }
                    }
                else:
                    yield {
                        "event": {
                            "type": "status",
                            "data": {"description": "Je lance une recherche dans les bases de connaissances disponibles...", "done": False, "hidden": False},
                        }
                    }
                
                useful_docs_lists = []
                for search in new_search_queries[:num_queries]:
                    useful_docs_lists.append(
                        SyncHelper.process_query_api(
                            client,
                            user_query,
                            search,
                            k,
                            log_messages,
                            collections_ids,
                        )
                    )

                # Flatten and deduplicate
                useful_docs = []
                seen = set()
                for doc_list in useful_docs_lists:
                    for doc in doc_list:
                        # Use a hash of the first 100 chars as a deduplication key
                        doc_hash = hash(doc[:100])
                        if doc_hash not in seen:
                            seen.add(doc_hash)
                            useful_docs.append(doc)

                logger.info(f"Useful docs after deduplication: {len(useful_docs)}")

                yield {
                    "event": {
                        "type": "status",
                        "data": {"description": f"J'ai trouvé {len(useful_docs)} documents pertinents...", "done": False, "hidden": False},
                    }
                }

                if (
                    useful_docs and num_queries > 1
                ):  # if we have more than one query it's complex, we need to process the docs
                    iteration_contexts = SyncHelper.process_useful_api_docs(
                        client,
                        user_query,
                        useful_docs,
                        log_messages,
                    )
                    aggregated_chunks.extend(useful_docs)
                else:
                    iteration_contexts = useful_docs
                    aggregated_chunks.extend(useful_docs)

                yield {
                    "event": {
                        "type": "status",
                        "data": {"description": "J'ai fini de lire les documents pertinents...", "done": False, "hidden": False},
                    }
                }

            if iteration_contexts:
                aggregated_contexts.extend(iteration_contexts)
                logger.info(f"Found {len(iteration_contexts)} useful contexts in iteration {iteration + 1}")
                log_messages.append(
                    f"Found {len(iteration_contexts)} useful contexts in iteration {iteration + 1}."
                )
            else:
                logger.info(f"No useful contexts found in iteration {iteration + 1}")
                log_messages.append(
                    f"No useful contexts found in iteration {iteration + 1}."
                )

            if iteration_limit > 1:
                new_search_queries = SyncHelper.get_new_search_queries(
                    client,
                    user_query,
                    all_search_queries,
                    aggregated_contexts,
                )
            else:
                new_search_queries = []

            if new_search_queries == "[]":
                logger.info("Research complete - no further searches needed")
                log_messages.append(
                    "LLM indicated that no further research is needed or iteration limit reached."
                )
                break
            elif new_search_queries:
                logger.info(f"Generated new search queries for next iteration: {new_search_queries}")
                log_messages.append(
                    f"New search queries for iteration {iteration + 2}: {new_search_queries}"
                )
                all_search_queries.extend(new_search_queries)
            else:
                logger.info("No new search queries provided - ending research")
                log_messages.append("No new search queries provided. Ending research.")
                break

            iteration += 1

        logger.info("=== GENERATING FINAL REPORT ===")
        log_messages.append("\nGenerating final report...")
        yield {
            "event": {
                "type": "status",
                "data": {"description": "Les recherches sont terminées ! Je résume les informations que j'ai trouvé...", "done": False, "hidden": False},
            }
        }

        final_prompt = SyncHelper.generate_final_prompt(
            client,
            user_query,
            aggregated_contexts,
            "",
            prompt_suffix,
            max_tokens,
        )

        logger.info(f"Final prompt: \n\n{final_prompt}\n\n")
        elapsed_time = time.time() - start_time

        logger.info("=== RESEARCH COMPLETED ===")
        logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
        logger.info(f"Total contexts found: {len(aggregated_contexts)}")

        log_messages.append(f"\nResearch completed in {elapsed_time:.2f} seconds.")

        # Citation for the front
        n_source = 0
        for chunk in aggregated_chunks:
            n_source += 1
            if internet:
                name = chunk
            else:
                name = f"Source {n_source}"
            yield {
                "event": {
                    "type": "citation",
                    "data": {
                        "document": [chunk],
                        "metadata": [
                            {
                                "source": f"[Source {n_source}]({extract_first_url(chunk)})"
                            }
                        ],
                        "source": {"name": name},
                    },
                }
            }

        # Make sure to yield the final prompt as the last item
        yield final_prompt

    except Exception as e:
        error_msg = f"Error during research: {str(e)}"
        logger.error(error_msg, exc_info=True)
        yield f"An error occurred: {str(e)}"


def stream_albert(
    client,
    model,
    max_tokens,
    messages,
    __event_emitter__,
    response_collector=None
):
    logger.info(f"=== STREAMING ALBERT RESPONSE ===")
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
                # Vérifie la structure des données comme dans l'exemple ci-dessus
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
                # On sort de la boucle si la génération est terminée
                if finish_reason is not None:
                    break

            except Exception as inner_e:
                logger.error(f"Error in streaming chunk: {inner_e}")
                continue

        # Stocker la réponse complète dans le collector si fourni
        if response_collector is not None:
            response_collector.append(output)

        # Dernier message : signaler la fin
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




class Pipeline:
    class Valves(BaseModel):
        max_turns: int = Field(
            default=4, description="Maximum allowable conversation turns for a user."
        )

        ALBERT_URL: str = Field(default="https://albert.api.etalab.gouv.fr/v1")
        ALBERT_KEY: str = Field(default="")
        MODEL: Literal[
            "albert-small",
            "albert-large",
        ] = Field(default="albert-large")
        RERANK_MODEL: str = Field(default="BAAI/bge-reranker-v2-m3")
        NUMBER_OF_CHUNKS: int = Field(default=5)
        SEARCH_SCORE_THRESHOLD: float = Field(default=0.35)
        RERANKER_SCORE_THRESHOLD: float = Field(default=0.1)
        REDACTOR_MODEL: str = Field(default="albert-small")
        ANALYTICS_MODEL: str = Field(default="albert-large")
        SUPERVISOR_MODEL: str = Field(default="albert-large")
        DEFAULT_MODEL: str = Field(default="albert-small")
        BRAVE_KEY: str = Field(default="")
        pass
 
    def __init__(self):
        self.name = PIPELINE_NAME
        self.valves = self.Valves()

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup: {__name__}")

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown: {__name__}")

    def pipe(
        self,
        body: dict,
        __event_emitter__=None,
        user_message=None,
        model_id=None,
        messages=None,
    )-> Union[str, Generator, Iterator]:
        logger.info(f"=== PIPE REQUEST STARTED ===")

        ALBERT_URL = self.valves.ALBERT_URL
        ALBERT_KEY = self.valves.ALBERT_KEY

        os.environ["REDACTOR_MODEL"] = self.valves.REDACTOR_MODEL
        os.environ["ANALYTICS_MODEL"] = self.valves.ANALYTICS_MODEL
        os.environ["SUPERVISOR_MODEL"] = self.valves.SUPERVISOR_MODEL
        os.environ["DEFAULT_MODEL"] = self.valves.DEFAULT_MODEL
        os.environ["COLLECTIONS"] = ""
        os.environ["ALBERT_URL"] = self.valves.ALBERT_URL
        os.environ["ALBERT_KEY"] = self.valves.ALBERT_KEY
        os.environ["BRAVE_KEY"] = self.valves.BRAVE_KEY
        model = self.valves.MODEL
        max_tokens = 4096
        PROMPT_COMPLEXE_OR_NOT_ADDON = ""
        PROMPT_SEARCH_ADDON = ""

        yield {
            "event": {
                "type": "status",
                "data": {
                    "description": "Je réfléchis...",
                    "done": False,
                },
            }
        }

        user_query = body.get("messages", [])[-1]["content"]
        messages = body.get("messages", [])

        logger.info(f"User query: {user_query}")

        client = OpenAI(
            api_key=ALBERT_KEY,
            base_url=self.valves.ALBERT_URL,
        )

        logger.info("=== EVALUATING SEARCH NECESSITY ===")
        search = client.chat.completions.create(
            model=model,
            stream=False,
            temperature=0.1,
            max_tokens=50,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT_SEARCH.format(
                        prompt_search_addon=PROMPT_SEARCH_ADDON,
                        history=messages[-10:],
                        question=user_query,
                    ),
                }
            ],
        )
        search = search.choices[0].message.content.strip().lower()
        logger.info(f"Search evaluation result: {search}")

        if search.strip().lower() != "no_search":
            logger.info("=== EVALUATING QUERY COMPLEXITY ===")
            complex_or_not = (
                client.chat.completions.create(
                    model=model,
                    stream=False,
                    temperature=0.1,
                    max_tokens=10,
                    messages=[
                        {
                            "role": "user",
                            "content": PROMPT_COMPLEXE_OR_NOT.format(
                                prompt_complex_or_not_addon=PROMPT_COMPLEXE_OR_NOT_ADDON,
                                history=messages[1:],
                                question=user_query,
                            ),
                        }
                    ],
                )
                .choices[0]
                .message.content
            )
            complex_or_not = complex_or_not.strip().lower()
            logger.info(f"Complexity evaluation: {complex_or_not}")

            if complex_or_not.strip().lower() == "complex":
                logger.info("=== HANDLING COMPLEX QUERY ===")
                iteration_limit = 5
                num_queries = 5
                k = 5
                prompt_suffix = ""
                search_type = "approndie"
            else:
                logger.info("=== HANDLING EASY QUERY ===")
                iteration_limit = 2
                num_queries = 1
                k = 3
                prompt_suffix = "Ignores les instructions précédentes, fais une réponse courte et concise qui répond à la question. L'utilisateur ne veut pas de réponse détaillée avec des informations inutiles."
                search_type = "rapide"


            research_generator = sync_research(
                client=client,
                user_query=search,
                internet=False if collections_ids else True,
                iteration_limit=iteration_limit,
                prompt_suffix=prompt_suffix,
                max_tokens=4096,
                num_queries=num_queries,
                k=k,
                collections_ids=collections_ids,
            )
            
            result = None
            for item in research_generator:
                if isinstance(item, dict) and "event" in item:
                    yield item
                else:
                    result = item

            messages = [
                {
                    "role": "user",
                    "content": result,
                },
            ]
            messages = body.get("messages", [])[-5:] + messages
            logger.info("Generating final response for complex query")
            logger.info(f"User prompt: {result}")

        else:
            logger.info("=== NO SEARCH REQUIRED ===")
            complex_or_not = "no_search"
            search = "no_search"
            messages = messages[-10:]


        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        logger.info(f"Messages: \n{messages}")

        for resp in stream_albert(client, model, max_tokens, messages, __event_emitter__):
            yield resp

        yield {
            "event": {
                "type": "status",
                "data": {
                    "description": "Réponse envoyée à l'utilisateur",
                    "done": True,
                    "hidden": True,
                },
            }
        }

        logger.info("=== PIPE REQUEST COMPLETED ===")
