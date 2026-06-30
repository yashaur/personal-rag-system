from app.config import settings

from langfuse.langchain import CallbackHandler
from langfuse import Langfuse, get_client

from langchain_core.callbacks import BaseCallbackHandler

import subprocess
import os
from time import perf_counter


def rag_version() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    except Exception:
        return 'Unknown'
    
Langfuse(
    host = settings.langfuse_base_url,
    public_key = settings.langfuse_public_key,
    secret_key = settings.langfuse_secret_key,
    release = rag_version()
)

def get_langfuse_handler(trace_id: str) -> CallbackHandler:
    return CallbackHandler(trace_context = {'trace_id': trace_id})


class OllamaLatencyCallback(BaseCallbackHandler):
    
    def __init__(self, label: str = 'unknown', trace_id: str = None):
        self.label = label
        self.trace_id = trace_id

    def on_chat_model_start(self, serialized, prompts, **kwargs):
        self._t_start = perf_counter()
        self._first_token_t = None

    def on_llm_new_token(self, token, **kwargs):
        if self._first_token_t is None:
            self._first_token_t = perf_counter()

    def on_llm_end(self, response, **kwargs):
        generation = response.generations[0][0]
        md = generation.message.response_metadata


        load_duration = md.get('load_duration', 0) / 1e9
        prompt_eval_count = md.get('prompt_eval_count', 0)
        prompt_eval_duration = md.get('prompt_eval_duration', 0) / 1e9
        eval_count = md.get('eval_count', 0)
        eval_duration = md.get('eval_duration', 0) / 1e9
        total_duration = md.get('total_duration', 0) / 1e9

        ttft = (self._first_token_t - self._t_start) if self._first_token_t else (load_duration + prompt_eval_duration)
        tpot = eval_duration / eval_count if eval_count > 0.0 else 0.0
        tps = eval_count / eval_duration if eval_duration > 0.0 else 0.0
        input_tokens = prompt_eval_count
        output_tokens = eval_count

        label = self.label

        client = get_client()

        if self.trace_id:
            client.create_score(name = f'{label}_ttft', value = ttft, data_type = 'NUMERIC', trace_id = self.trace_id)
            client.create_score(name = f'{label}_tpot', value = tpot, data_type = 'NUMERIC', trace_id = self.trace_id)
            client.create_score(name = f'{label}_tps', value = tps, data_type = 'NUMERIC', trace_id = self.trace_id)
            client.create_score(name = f'{label}_load_duration', value = load_duration, data_type = 'NUMERIC', trace_id = self.trace_id)
            client.create_score(name = f'{label}_total_duration', value = total_duration, data_type = 'NUMERIC', trace_id = self.trace_id)
            client.create_score(name = f'{label}_input_tokens', value = input_tokens, data_type = 'NUMERIC', trace_id = self.trace_id)
            client.create_score(name = f'{label}_output_tokens', value = output_tokens, data_type = 'NUMERIC', trace_id = self.trace_id)

        else:
            client.score_current_trace(name = f'{label}_ttft', value = ttft, data_type = 'NUMERIC')
            client.score_current_trace(name = f'{label}_tpot', value = tpot, data_type = 'NUMERIC')
            client.score_current_trace(name = f'{label}_tps', value = tps, data_type = 'NUMERIC')
            client.score_current_trace(name = f'{label}_load_duration', value = load_duration, data_type = 'NUMERIC')
            client.score_current_trace(name = f'{label}_total_duration', value = total_duration, data_type = 'NUMERIC')
            client.score_current_trace(name = f'{label}_input_tokens', value = input_tokens, data_type = 'NUMERIC')
            client.score_current_trace(name = f'{label}_output_tokens', value = output_tokens, data_type = 'NUMERIC')


if __name__ == '__main__':
    print(rag_version())