import chainlit as cl
from src.structured_output import structured_pipeline
from src.vectorDatabase import create_database
from src.queryDatabase import query_database_earnings_call,query_database_sec
from sentence_transformers import SentenceTransformer
from src.config import *
import torch
from src.chat_earnings_call import get_openai_answer_earnings_call
from src.chat_sec import get_openai_answer_sec

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = SentenceTransformer(
        ENCODER_NAME, device=device, trust_remote_code=True
    )


@cl.on_chat_start
def on_chat_start():
    print("A new chat session has started")

@cl.on_message
async def on_message(msg: cl.Message):
    query =  structured_pipeline(msg.content)
    # await cl.Message(ans).send()

    tickers = query.question_ticker_quarter_year.ticker
    years = query.question_ticker_quarter_year.year
    quarter = query.question_ticker_quarter_year.quarter

    tic_yr_dict = {}
    for tic in tickers:
        for yr in years:
            if f"qdrant_client_{tic}_{yr}" in tic_yr_dict: continue
            print(f"Building vector database for {tic} and year {yr}")
            qdrant_client,speakers_list_1,speakers_list_2,speakers_list_3,speakers_list_4,sec_form_names,earnings_call_quarter_vals = create_database(tic,yr)
            qd_client_var_name = f"qdrant_client_{tic}_{yr}"
            tic_yr_dict[qd_client_var_name] = qdrant_client
            speakers_list_1_var_name = f"speakers_list_1_{tic}_{yr}"
            tic_yr_dict[speakers_list_1_var_name] = speakers_list_1
            speakers_list_2_var_name = f"speakers_list_2_{tic}_{yr}"
            tic_yr_dict[speakers_list_2_var_name] = speakers_list_2
            speakers_list_3_var_name = f"speakers_list_3_{tic}_{yr}"
            tic_yr_dict[speakers_list_3_var_name] = speakers_list_3
            speakers_list_4_var_name = f"speakers_list_4_{tic}_{yr}"
            tic_yr_dict[speakers_list_4_var_name] = speakers_list_4
            sec_form_names_var_name = f"sec_form_names_{tic}_{yr}"
            tic_yr_dict[sec_form_names_var_name] = sec_form_names
            earnings_call_quarter_vals_var_name = f"earnings_call_quarter_vals_{tic}_{yr}"
            tic_yr_dict[earnings_call_quarter_vals_var_name] = earnings_call_quarter_vals
        print(f"Done for {tic} and year {yr}")
    
    question = query.rewritten_query
    source = query.question_ticker_quarter_year.data_source
    relevant_text = ""
    for tic in tickers:
        if source == "CALLS":
            for q in quarter:
                print(q)
                if q == "Q1":
                    speakers_list = tic_yr_dict[f"speakers_list_1_{tic}_{yr}"]
                elif q == "Q2":
                    speakers_list = tic_yr_dict[f"speakers_list_2_{tic}_{yr}"]
                elif q == "Q3":
                    speakers_list = tic_yr_dict[f"speakers_list_3_{tic}_{yr}"]
                elif q == "Q4":
                    speakers_list = tic_yr_dict[f"speakers_list_4_{tic}_{yr}"]
                
                relevant_text += f"For {tic} and Quarter {q}\n"
                relevant_text += query_database_earnings_call(question,q,tic_yr_dict[f"qdrant_client_{tic}_{yr}"],encoder,speakers_list)
          
        if source == "SEC":
            if quarter == [""]:
                search_form = "10-K"
                relevant_text += query_database_sec(question, tic_yr_dict[f"qdrant_client_{tic}_{yr}"], encoder, search_form)
            else:
                for q in quarter:
                    search_form = "10-"+q
                    relevant_text += f"For ticker {tic} and Quarter {q}\n"
                    relevant_text += query_database_sec(question, tic_yr_dict[f"qdrant_client_{tic}_{yr}"], encoder, search_form)

    if source == 'CALLS':
        chat_msg = get_openai_answer_earnings_call(question,relevant_text)
    elif source == 'SEC':
        chat_msg = get_openai_answer_sec(question,relevant_text)
    
    await cl.Message(chat_msg).send()