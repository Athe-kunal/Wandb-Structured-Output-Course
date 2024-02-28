import instructor

from openai import AsyncOpenAI
from typing import List, Literal
from enum import Enum
from pydantic import AfterValidator, Field, BaseModel
from datetime import datetime
import pandas as pd
from instructor import llm_validator
from typing_extensions import Annotated


class TickerYearQuarter(BaseModel):
    chain_of_thought: str = Field(
        description="Think step by step to output what is the ticker symbols, NOT THE COMPANY NAME, quarter, year and data source the question is talking about"
    )
    ticker:List[str] 
    year: List[str] = Field(description="The year that the question is talking about")
    quarter: List[str] = Field(description="The quarter number that the question is talking about. Make sure that it starts with Q, for example Quarter 4 is Q4")
    data_source: Literal["CALLS","SEC"] = Field(description="If the question is talking about SEC filings then output SEC, else if the question is talking about Earning calls transcript then output CALLS")

class Query(BaseModel):
    rewritten_query: str = Field(description="Rewrite the query and DON'T include the company name, years, quarters and data sources")
    question_ticker_quarter_year: TickerYearQuarter


aclient = instructor.patch(AsyncOpenAI())

FinanceTopicStr = Annotated[
    str,
    AfterValidator(
        llm_validator(
            "don't talk about any other topic except finance",
            openai_client=aclient,
        )
    ),
]

class AssistantMessage(BaseModel):
    message: FinanceTopicStr

def expand_query(q) -> Query:
    datetime_obj =  datetime.today().strftime("%Y-%m-%d")
    quarter = pd.Timestamp(datetime_obj).quarter
    year = pd.Timestamp(datetime_obj).year
    return aclient.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        temperature=0.0,
        response_model=Query,
        messages=[
            {
                "role": "system",
                "content": f"You're a query understanding system for SEC Filings and Earnings Call. The current year is {year} and quarter {quarter}. Here are some tips: ...",
            },
            {"role": "user", "content": f"query: {q}"},
        ],
    )

def structured_pipeline(question):
    try:
        AssistantMessage(
            message=question
        )
        return expand_query(question)
    except AttributeError as e:
        print("This is a financial chatbot, please talk about it financial related topics")