import torch
from openai import OpenAI
from fastchat.model import load_model, get_conversation_template
import logging
import time
import concurrent.futures
from vllm import LLM as vllm
from vllm import SamplingParams
import google.generativeai as palm
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import ollama
from together import Together
import requests


class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def generate(self, prompt):
        raise NotImplementedError("LLM must implement generate method.")

    def predict(self, sequences):
        raise NotImplementedError("LLM must implement predict method.")


class LocalLLM(LLM):
    def __init__(self,
                 model_path,
                 device='cuda',
                 num_gpus=1,
                 max_gpu_memory=None,
                 dtype=torch.float16,
                 load_8bit=False,
                 cpu_offloading=False,
                 revision=None,
                 debug=False,
                 system_message=None
                 ):
        super().__init__()

        self.model, self.tokenizer = self.create_model(
            model_path,
            device,
            num_gpus,
            max_gpu_memory,
            dtype,
            load_8bit,
            cpu_offloading,
            revision=revision,
            debug=debug,
        )
        self.model_path = model_path

        if system_message is None and 'Llama-2' in model_path:
            self.system_message =  "You are a helpful code assistant that can teach a junior developer how to code. " \
            "Don't explain the code, just generate the code block itself. \n" \
            "Write code for the following programming task:"
        else:
            self.system_message = system_message

    @torch.inference_mode()
    def create_model(self, model_path,
                     device='cuda',
                     num_gpus=1,
                     max_gpu_memory=None,
                     dtype=torch.float16,
                     load_8bit=False,
                     cpu_offloading=False,
                     revision=None,
                     debug=False):
        model, tokenizer = load_model(
            model_path,
            device,
            num_gpus,
            max_gpu_memory,
            dtype,
            load_8bit,
            cpu_offloading,
            revision=revision,
            debug=debug,
        )

        return model, tokenizer

    def set_system_message(self, conv_temp):
        if self.system_message is not None:
            conv_temp.set_system_message(self.system_message)

    @torch.inference_mode()
    def generate(self, prompt, temperature=0.0, max_tokens=4096, repetition_penalty=1.0):
        conv_temp = get_conversation_template(self.model_path)
        self.set_system_message(conv_temp)

        conv_temp.append_message(conv_temp.roles[0], prompt)
        conv_temp.append_message(conv_temp.roles[1], None)

        prompt_input = conv_temp.get_prompt()
        input_ids = self.tokenizer([prompt_input]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=False,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_tokens
        )

        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]):]

        return self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )

    @torch.inference_mode()
    def generate_batch(self, prompts, temperature=0.0, max_tokens=4096, repetition_penalty=1.0, batch_size=16):
        prompt_inputs = []
        for prompt in prompts:
            conv_temp = get_conversation_template(self.model_path)
            self.set_system_message(conv_temp)

            conv_temp.append_message(conv_temp.roles[0], prompt)
            conv_temp.append_message(conv_temp.roles[1], None)

            prompt_input = conv_temp.get_prompt()
            prompt_inputs.append(prompt_input)

        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        input_ids = self.tokenizer(prompt_inputs, padding=True).input_ids
        # load the input_ids batch by batch to avoid OOM
        outputs = []
        for i in range(0, len(input_ids), batch_size):
            output_ids = self.model.generate(
                torch.as_tensor(input_ids[i:i+batch_size]).cuda(),
                do_sample=False,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_tokens,
            )
            output_ids = output_ids[:, len(input_ids[0]):]
            outputs.extend(self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True, spaces_between_special_tokens=False))
        return outputs


class LocalOllamaLLM(LLM):
    def __init__(self, model_name: str = "codellama:7b", system_message=None):
        """
        :param model_name: Model to use, e.g, 'llama3.1'
        """
        self.model_name = model_name
        self.system_message = system_message if system_message is not None else (
            "You are a helpful code assistant that can teach a junior developer how to code. " \
            "Don't explain the code, just generate the code block itself. \n" \
            "Write code for the following programming task:"
        )

    def get_model_name(self):
        return self.model_name

    def generate(self, prompt: str):
        try:
            messages = [
                {'role': 'system', 'content': self.system_message},
                {'role': 'user', 'content': prompt}
            ]
            response = ollama.chat(model=self.model_name, messages=messages)
            return response['message']['content']
        except ollama.ResponseError as e:
            print(f"Error occurred: {e.error}")
            return None


    def generate_batch(self, prompts, temperature=0.7, max_tokens=512):
        generated_responses = []

        for prompt in prompts:
            response = self.generate(
                prompt=prompt
            )
            generated_responses.append(response)

        return generated_responses


class LocalVLLM(LLM):
    def __init__(self,
                 model_path,
                 gpu_memory_utilization=0.95,
                 system_message=None
                 ):
        super().__init__()
        self.model_path = model_path
        self.model = vllm(
            self.model_path, gpu_memory_utilization=gpu_memory_utilization)
        
        if system_message is None and 'Llama-2' in model_path:
            self.system_message = (
                "You are a helpful code assistant that can teach a junior developer how to code. " \
                "Don't explain the code, just generate the code block itself. \n" \
                "Write code for the following programming task:"
            )
        else:
            self.system_message = system_message

    def set_system_message(self, conv_temp):
        if self.system_message is not None:
            conv_temp.set_system_message(self.system_message)

    def generate(self, prompt, temperature=0, max_tokens=512):
        prompts = [prompt]
        return self.generate_batch(prompts, temperature, max_tokens)

    def generate_batch(self, prompts, temperature=0, max_tokens=512):
        prompt_inputs = []
        for prompt in prompts:
            conv_temp = get_conversation_template(self.model_path)
            self.set_system_message(conv_temp)

            conv_temp.append_message(conv_temp.roles[0], prompt)
            conv_temp.append_message(conv_temp.roles[1], None)

            prompt_input = conv_temp.get_prompt()
            prompt_inputs.append(prompt_input)

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        results = self.model.generate(
            prompt_inputs, sampling_params, use_tqdm=False)
        outputs = []
        for result in results:
            outputs.append(result.outputs[0].text)
        return outputs


class BardLLM(LLM):
    def generate(self, prompt):
        return

class PaLM2LLM(LLM):
    def __init__(self,
                 model_path='chat-bison-001',
                 api_key=None,
                 system_message=None
                ):
        super().__init__()
        
        if len(api_key) != 39:
            raise ValueError('invalid PaLM2 API key')
        
        palm.configure(api_key=api_key)
        available_models = [m for m in palm.list_models()]
        for model in available_models:
            if model.name == model_path:
                self.model_path = model
                break
        self.system_message = system_message
        # The PaLM-2 has a great rescriction on the number of input tokens, so I will release the short jailbreak prompts later
        
    def generate(self, prompt, temperature=0, n=1, max_trials=1, failure_sleep_time=1):
        for _ in range(max_trials):
            try:
                results = palm.chat(
                    model=self.model_path,
                    prompt=prompt,
                    temperature=temperature,
                    candidate_count=n,
                )
                return [results.candidates[i]['content'] for i in range(n)]
            except Exception as e:
                logging.warning(
                    f"PaLM2 API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" " for _ in range(n)]
    
    def generate_batch(self, prompts, temperature=0, n=1, max_trials=1, failure_sleep_time=1):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, n,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results

class ClaudeLLM(LLM):
    def __init__(self,
                 model_path='claude-instant-1.2',
                 api_key=None
                ):
        super().__init__()
        
        if len(api_key) != 108:
            raise ValueError('invalid Claude API key')
        
        self.model_path = model_path
        self.api_key = api_key
        self.anthropic = Anthropic(
            api_key=self.api_key
        )

    def generate(self, prompt, max_tokens=1024, max_trials=1, failure_sleep_time=1):
        
        for _ in range(max_trials):
            try:
                completion = self.anthropic.completions.create(
                    model=self.model_path,
                    max_tokens_to_sample=300,
                    prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
                )
                return [completion.completion]
            except Exception as e:
                logging.warning(
                    f"Claude API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" "]
    
    def generate_batch(self, prompts, max_tokens=1024, max_trials=1, failure_sleep_time=1):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, max_tokens,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results

class OpenAILLM(LLM):
    def __init__(self,
                 model_path,
                 api_key=None,
                 system_message=None
                ):
        super().__init__()

        if not api_key.startswith('sk-'):
            raise ValueError('OpenAI API key should start with sk-')
        if model_path not in ['gpt-3.5-turbo', 'gpt-4']:
            raise ValueError(
                'OpenAI model path should be gpt-3.5-turbo or gpt-4')
        self.client = OpenAI(api_key = api_key)
        self.model_path = model_path
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."

    def generate(self, prompt, temperature=0, max_tokens=1024, n=1, max_trials=10, failure_sleep_time=5):
        for _ in range(max_trials):
            try:
                results = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )
                return [results.choices[i].message.content for i in range(n)]
            except Exception as e:
                logging.warning(
                    f"OpenAI API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" " for _ in range(n)]

    def generate_batch(self, prompts, temperature=0, max_tokens=1024, n=1, max_trials=10, failure_sleep_time=5):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, max_tokens, n,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results
    
class DeepSeekLLM(LLM):
    def __init__(self,
                    model_path,
                    api_key=None,
                    system_message=None
                ):
        super().__init__()

        if not api_key.startswith('sk-'):
            raise ValueError('DeepSeek API key should start with sk-')
        if model_path not in ['deepseek-chat', 'deepseek-coder']:
            raise ValueError(
                'DeepSeek model path should be deepseek-chat or deepseek-coder')
        self.client = OpenAI(api_key = api_key, base_url="https://api.deepseek.com/v1")
        self.model_path = model_path
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."

    def generate(self, prompt, temperature=0, max_tokens=1024, n=1, max_trials=10, failure_sleep_time=5):
        for _ in range(max_trials):
            try:
                results = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )
                return [results.choices[i].message.content for i in range(n)]
            except Exception as e:
                logging.warning(
                    f"DeepSeek API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" " for _ in range(n)]

    def generate_batch(self, prompts, temperature=0, max_tokens=1024, n=1, max_trials=10, failure_sleep_time=5):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, max_tokens, n,
                                        max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results

class QwenLLM(LLM):
    def __init__(self,
                    model_path,
                    api_key=None,
                    system_message=None,
                    temperature=0.2
                ):
        super().__init__()

        if not api_key.startswith('sk-'):
            raise ValueError('Qwen API key should start with sk-')

        self.client = OpenAI(api_key = api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model_path = model_path
        self.temperature = temperature 
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."

    def generate(self, prompt, temperature=0.2, max_tokens=4096, n=1, max_trials=10, failure_sleep_time=5):
        for _ in range(max_trials):
            try:
                results = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )
                return [results.choices[i].message.content for i in range(n)]
            except Exception as e:
                logging.warning(
                    f"API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" " for _ in range(n)]

    def generate_batch(self, prompts, temperature=0.2, max_tokens=4096, n=1, max_trials=10, failure_sleep_time=5):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, max_tokens, n,
                                        max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results


class SiliconFlowLLM(LLM):
    def __init__(self,
                 model_path="deepseek-ai/DeepSeek-V3",
                 api_key=None,
                 system_message=None
                ):
        super().__init__()

        if api_key is None or not api_key.strip():
            raise ValueError('SiliconFlow API key must be provided')
        
        self.api_key = api_key
        self.model_path = model_path
        self.base_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."
        
    def generate(self, prompt, temperature=0.2, max_tokens=4096, n=1, top_p=0.95, 
                 top_k=50, frequency_penalty=0.5, max_trials=10, failure_sleep_time=5):
        """
        Generate text using SiliconFlow API.
        
        Args:
            prompt: The input prompt
            temperature: Controls randomness (higher = more random)
            max_tokens: Maximum number of tokens to generate
            n: Number of responses to generate
            top_p: Nucleus sampling parameter
            top_k: Number of highest probability tokens to consider
            frequency_penalty: Penalty for token frequency
            max_trials: Maximum number of retry attempts
            failure_sleep_time: Seconds to wait between retries
            
        Returns:
            List of generated responses
        """
        messages = [{"role": "user", "content": prompt}]
        
        # Add system message if available
        if self.system_message:
            messages = [{"role": "system", "content": self.system_message}] + messages
            
        payload = {
            "model": self.model_path,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "stop": None,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "n": n,
            "response_format": {"type": "text"}
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        for attempt in range(max_trials):
            try:
                response = requests.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()  # Raise exception for HTTP errors
                
                result = response.json()
                return [choice["message"]["content"] for choice in result.get("choices", [])]
                
            except Exception as e:
                logging.warning(
                    f"SiliconFlow API call failed due to {e}. Retrying {attempt+1}/{max_trials}...")
                if attempt < max_trials - 1:
                    time.sleep(failure_sleep_time)
        
        # Return empty strings if all attempts failed
        return [" " for _ in range(n)]
    
    def generate_batch(self, prompts, temperature=0.2, max_tokens=4096, n=1, 
                      top_p=0.95, top_k=50, frequency_penalty=0.5,
                      max_trials=10, failure_sleep_time=5):
        """Generate responses for a batch of prompts using parallel processing"""
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.generate, 
                    prompt,
                    temperature, 
                    max_tokens, 
                    n,
                    top_p, 
                    top_k, 
                    frequency_penalty,
                    max_trials, 
                    failure_sleep_time
                ): prompt for prompt in prompts
            }
            
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
                
        return results

class TogetherLLM(LLM):
    def __init__(self, model_path, api_key=None, system_message=None):
        super().__init__()
        
        if api_key is None:
            raise ValueError("API key must be provided either directly or through the TOGETHER_API_KEY environment variable.")
        
        self.client = Together(api_key=api_key)
        self.model_path = model_path
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."

    def generate(self, prompt, temperature=0.2, max_tokens=4096, n=1, max_trials=10, failure_sleep_time=5):
        for _ in range(max_trials):
            try:
                results = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )
                return [results.choices[i].message.content for i in range(n)]
            except Exception as e:
                logging.warning(
                    f"Together API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" " for _ in range(n)]

    def generate_batch(self, prompts, temperature=0.2, max_tokens=4096, n=1, max_trials=10, failure_sleep_time=5):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, max_tokens, n,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results