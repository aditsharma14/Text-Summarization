import os
import validators,streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

## Load .env from the project folder, regardless of where streamlit is launched from
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)),".env"))

st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')



## Get the Groq API Key and url(YT or website)to be summarized
## (pre-filled from `groq_api` in .env if present)
with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value=os.getenv("groq_api",""),type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")

## Gemma-7b-It is no longer available on Groq, so use a currently supported model
GROQ_MODEL="openai/gpt-oss-20b"
## Groq free tier allows ~8000 tokens/minute, so very long pages are trimmed to the first few chunks
MAX_CHUNKS=4

prompt_template="""
Provide a summary of the following content in about 300 words:
Content:{text}

"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

## Prompts used when the content is too long to send in a single request
map_prompt=PromptTemplate(template="Summarize the following part of the content:\n{text}\nSummary:",
                          input_variables=["text"])

YOUTUBE_HOSTS=("youtube.com","youtu.be")

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")

    else:
        try:
            with st.spinner("Waiting..."):
                ## The LLM is created here so an empty API key doesn't crash the app on startup
                ## reasoning_effort="low" stops the reasoning model from using its whole output budget
                ## on thinking (e.g. counting words); max_retries waits out Groq rate limits
                llm=ChatGroq(model=GROQ_MODEL, api_key=groq_api_key, reasoning_effort="low", max_retries=10)

                ## loading the website or yt video data
                if any(host in generic_url for host in YOUTUBE_HOSTS):
                    ## add_video_info relies on pytube, which is broken against current YouTube
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=False)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs=loader.load()

                if not docs or not any(doc.page_content.strip() for doc in docs):
                    st.error("Could not extract any text from that URL.")
                else:
                    ## Chain For Summarization: "stuff" for short content, "map_reduce" for long content
                    docs=RecursiveCharacterTextSplitter(chunk_size=6000,chunk_overlap=200).split_documents(docs)
                    if len(docs)>MAX_CHUNKS:
                        st.info(f"Content is long, so only the first {MAX_CHUNKS} of {len(docs)} sections are summarized.")
                        docs=docs[:MAX_CHUNKS]
                    if len(docs)==1:
                        chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                    else:
                        chain=load_summarize_chain(llm,chain_type="map_reduce",
                                                   map_prompt=map_prompt,combine_prompt=prompt)
                    output_summary=chain.invoke({"input_documents":docs})["output_text"]

                    st.success(output_summary)
        except Exception as e:
            st.exception(e)
