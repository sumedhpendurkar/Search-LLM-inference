import os
import openai
import numpy as np
from typing import Optional, Union, Literal
import time

from reasoners import LanguageModel, GenerateOutput
import tiktoken
from openai import OpenAI

PROMPT_TEMPLATE_ANSWER = "Your response need to be ended with \"So the answer is\"\n\n"
PROMPT_TEMPLATE_CONTINUE = "Please continue to answer the last question, following the format of previous examples. Don't say any other words.\n\n"

class OpenAIModel(LanguageModel):
    def __init__(self, model:str, max_tokens:int = 2048, temperature=0.0, additional_prompt=None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY", None),
            # organization='',
        )
        self.additional_prompt = additional_prompt
    
    def generate(self,
                prompt: Optional[Union[str, list[str]]],
                max_tokens: int = None,
                top_p: float = 1.0,
                num_return_sequences: int = 1,
                rate_limit_per_min: Optional[int] = 20,
                eos_token_id: Optional[str] = None,
                logprobs: Optional[int] = None,
                temperature = None,
                additional_prompt=None,
                retry = 64,
                **kwargs) -> GenerateOutput:
       
        stop = eos_token_id
        gpt_temperature = self.temperature if temperature is None else temperature
        if isinstance(prompt, list):
            assert len(prompt) == 1
            prompt = prompt[0]
        if additional_prompt is None and self.additional_prompt is not None:
            additional_prompt = self.additional_prompt
        elif additional_prompt is not None and self.additional_prompt is not None:
            print("Warning: additional_prompt set in constructor is overridden.")
        if additional_prompt == "ANSWER":
            prompt = PROMPT_TEMPLATE_ANSWER + prompt
        elif additional_prompt == "CONTINUE":
            prompt = PROMPT_TEMPLATE_CONTINUE + prompt

        if max_tokens is None:
            max_tokens = self.max_tokens
        
        if logprobs is None:
            logprobs = 0


        for i in range(1, retry + 1):
            try:
                # sleep several seconds to avoid rate limit
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)
                ### GPT 3.5 and higher use a different API
                if ('gpt-3.5' in self.model) or ('gpt-4' in self.model):
                    messages = [{"role": "user", "content": prompt}]
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        stop=stop
                    )
                    return GenerateOutput(
                        text=[choice.message.content for choice in response.choices],
                        log_prob=None
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        stop=stop,
                        logprobs=0,
                        **kwargs
                    )
                    return GenerateOutput(
                        text=[choice["text"] for choice in response.choices],
                        log_prob=[choice["logprobs"] for choice in response["choices"]]
                    )
            
            except Exception as e:
                print(f"An Error Occured: {e}, sleeping for {i} seconds")
                time.sleep(i)
        
        # after 64 tries, still no luck
        raise RuntimeError("GPTCompletionModel failed to generate output, even after 64 tries")
    
    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]],
                              **kwargs) -> list[np.ndarray]:
        """
        Compute logits for the next token given a prompt and candidates, using OpenAI API.

        Args:
            prompt (Union[str, list[str]]): The input prompt as a string or list of strings.
            candidates (Union[list[str], list[list[str]]]): A list of candidate tokens or token sequences.

        Returns:
            list[np.ndarray]: A list of numpy arrays containing the logits for each candidate token.
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(candidates[0], str):
            candidates = [candidates] * len(prompt)

        results = []
        for p, cands in zip(prompt, candidates):
            # Construct the messages for OpenAI API
            messages = [{"role": "user", "content": p}]

            # Call OpenAI API to get logprobs
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1,  # We only care about the next token
                temperature=0,  # Deterministic output
                logprobs=True,  # Request top logits
                top_logprobs=10,
                **kwargs
            )

            choices = response.choices[0]
            logprobs = choices.logprobs.content  # logprobs is a list
            top_logprobs = logprobs[0].top_logprobs  # List of items with .token and .logprob

            cand_logits = []
            for cand in cands:
                # Try to find the candidate in top_logprobs
                matching_items = [item for item in top_logprobs if item.token == cand]
                if matching_items:
                    # Candidate found, append its logprob
                    cand_logits.append(matching_items[0].logprob)
                else:
                    # Candidate not found, use minimum logprob
                    cand_logits.append(min(item.logprob for item in top_logprobs))

            results.append(np.array(cand_logits, dtype=np.float32))

        return results

    def get_loglikelihood(self,
                          prefix: str,
                          contents: list[str],
                          temperature=1) -> list[np.ndarray]:
      
        tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
        contents_tokens = [tokenizer.encode(content[len(prefix):]) for content in contents]
        #max_contents_token_len = max([len(tokens) for tokens in contents_token])
   
        acc_probs = np.zeros(len(contents), dtype=np.float32)

        for j in range(len(contents_tokens)):
            for i in range(len(contents_tokens[j])):
                # i should be max over contents
                completion = self.client.chat.completions.create(
                  model=self.model, logprobs=True, top_logprobs=10,
                  messages=[
                      {"role": "user", "content": prefix + tokenizer.decode(contents_tokens[j][:i])}
                  ],
                  temperature=temperature,
                  max_tokens=1
                )

                # Extract the response
                choices = completion.choices[0]
                logprobs = choices.logprobs.content  # This will contain the top token options for each token
                try: 
                    acc_probs[j] += next(item.logprob for item in logprobs[0].top_logprobs \
                            if item.token == tokenizer.decode([contents_tokens[j][i]]))
                except StopIteration:
                    print("Word not found")
                    acc_probs[j] += min(item.logprob for item in logprobs[0].top_logprobs) 
                #print(logprobs[0].top_logprobs[:3], list(tokenizer.decode([contents_tokens[j][i]])), acc_probs)
        return acc_probs


if __name__ == '__main__':
    model = OpenAIModel(model='gpt-3.5-turbo')
    print(model.generate(['The capital of UK is']))
    print(model.get_next_token_logits(["The capital of UK is", "The capital of France is", "The capital of Russia is"], ["Paris", "London", "Moscow"]))
    #print(np.exp(model.get_loglikelihood("The capital of UK is ", ["The capital of UK is Paris.", "The capital of UK is London.", "The capital of UK is Moscow."])))
