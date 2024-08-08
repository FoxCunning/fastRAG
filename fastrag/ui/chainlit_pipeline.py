import argparse
import base64
import json
import os

import chainlit as cl
from chainlit.sync import run_sync
from events import Events
from haystack.components.generators.hugging_face_local import HuggingFaceLocalGenerator

from fastrag.agents.base import AgentTokenStreamingHandler, HFTokenStreamingHandler
from fastrag.agents.create_agent import get_basic_conversation_pipeline

from pypdf import PdfReader

config = os.environ.get("CONFIG", "config/rag_pipeline_chat.yaml")

args = argparse.Namespace(app_type="conversation", config=config)

generator, tools_objects = get_basic_conversation_pipeline(args)
callback_manager = Events()

stream_handler = AgentTokenStreamingHandler(callback_manager)
generator.generation_kwargs["streamer"] = HFTokenStreamingHandler(
    generator.pipeline.tokenizer, stream_handler
)

memory = [
    {
        "role": "system",
        "content": """You are a helpful assistant.
Your answers must be short and to the point.
""",
    }
]


def extract_bookmarks(reader, bookmarks) -> list:
    """Recursively extract all bookmark titles and starting pages."""
    output = []
    for bm in bookmarks:
        print(".", end=".")
        if type(bm) == list:
            output = output + extract_bookmarks(reader, bm)
        else:
            title = bm["/Title"]
            start = reader.get_destination_page_number(bm)
            output.append({"title": title, "first": start, "last": -1})
        return output


def ingest_pdf(path: str) -> list:
    """Convert a PDF to a list of dictionaries containing title
    and plain text for each chapter, using bookmarks."""
    data = []
    try:
        reader = PdfReader(path)
        bookmarks = reader.outline
        
        if bookmarks is None or len(bookmarks) == 0:
            # No bookmarks found: import all text as one item
            
            # Try to extract title
            if reader.metadata.title is None:
                if reader.metadata.subject is None:
                # If no title and no subject, use file name
                    title = os.path.basename(element.path).split('.')[0]
                else:
                    # If no title, use subject
                    title = reader.metadata.subject
            else:
                # Use title from metadata
                title = reader.metadata.title

            chapter = {"image_url": "#", "title": title, "content": ""}
            for page in reader.pages:
                chapter["content"] = chapter["content"] + "\n" + page.extract_text()
                
            data = [chapter]
            
        else:
            # Use bookmarks to extract each chapter's start page
            print("Processing bookmarks", end="")
            bm_list = extract_bookmarks(reader, bookmarks)
            
            # Each chapter's last page is the next chapter's first page,
            # except for the last chapter
            for b in range(len(bm_list) - 1):
                bm_list[b]["last"] = bm_list[b + 1]["first"]
            bm_list[-1]["last"] = len(reader.pages) - 1
            
            # Extract each chapter's text
            print("\nExtracting text", end="")
            for bm in bm_list:
                print(".", end="")
                start = bm["first"]
                end = bm["last"]
                
                for p in range(start, end + 1):
                    print(".", end="")
                    rows = reader.pages[p].extract_text().splitlines(keepends=True) # [1:] this would skip the header
                    # Skip page number
                    pos = rows[1].find(" ", 3) + 1 # Change index to 0 if skipping header
                    rows[1] = rows[1][pos:]
                    # Add this page to the output
                    data.append({"image_url": "#", "title": bm["title"], "content": "".join(rows)})
            print(".")
            
    except OSError as err:
        print(f"!!!ERROR: While converting '{path}': {err}")
        return []

    return data


@cl.on_chat_end
def chat_end():
    global memory
    # clear memory
    memory = []


@cl.on_message
async def main(message: cl.Message):
    global current_settings

    def parse_element(element, params):
        # insert the image into the params
        if "image" in element.mime:
            img_str = base64.b64encode(element.content).decode("utf-8")
            if "images" not in params:
                params["images"] = []
            params["images"].append(img_str)

        # insert the text into the params
        if "text" in element.mime:
            if element.content is None:
                # print(f"\n*** Element ({element.mime}) has no content!")
                # print(f"\tName: {element.name}\n\tURL: {element.url}\n\tPath: {element.path}\n")
                with open(element.path, "rb") as fd:
                    element.content = fd.read()

            file_text = element.content.decode("utf-8")
            if "file_texts" not in params:
                params["file_texts"] = []
            params["file_texts"].append(file_text.split("\n"))

        if "pdf" in element.mime:
            if "data_rows" not in params:
                params["data_rows"] = []
            
            print(f"--- Reading PDF from: {element.path}")
            data = ingest_pdf(element.path)
            if len(data) > 0:
                print(f"--- Extracted {len(data)} chapters from PDF.")
                params["data_rows"].append(data)
                

        if "json" in element.mime:
            if "data_rows" not in params:
                params["data_rows"] = []

            data_rows = json.load(open(element.path, "r"))
            params["data_rows"].append(data_rows)

    # params for the agent
    params = {}

    if len(message.elements) > 0:
        # parse the input elements, such as images, text, etc.
        for el in message.elements:
            parse_element(el, params)

        # Upload text and images, when appropriate
        for tool in tools_objects:
            if hasattr(tool, "upload_data_to_pipeline"):
                _ = await cl.Message(
                    author="Agent", content=f"Uploading into {tool.name}..."
                ).send()
                tool.upload_data_to_pipeline(params)

    user_query = message.content
    user_input = user_query
    for tool in tools_objects:
        tool_result = tool.run(user_query)
        user_input += tool_result
        _ = await cl.Message(author=tool.name, content=tool_result).send()

    memory.append({"role": "user", "content": user_input})

    prompt = generator.pipeline.tokenizer.apply_chat_template(
        memory, tokenize=False, add_generation_prompt=True
    )

    message = cl.Message(author="Agent", content="")

    def stream_to_message(token):
        run_sync(message.stream_token(token))

    callback_manager.on_new_token = stream_to_message

    result = await cl.make_async(generator.run)(prompt)

    answer = result["replies"][0]

    memory.append({"role": "assistant", "content": answer})
